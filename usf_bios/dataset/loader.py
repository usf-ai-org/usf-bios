# Copyright (c) US Inc. All rights reserved.
import os
from contextlib import nullcontext
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from datasets import Dataset as HfDataset
from datasets import load_dataset as hf_load_dataset
from modelscope.hub.utils.utils import get_cache_dir

from usf_bios.hub import get_hub
from usf_bios.utils import get_logger, get_seed, safe_ddp_context, use_hf_hub
from usf_bios.system_guard import validate_dataset_source, DatasetSourceError
from .dataset_meta import DATASET_TYPE, BaseDatasetLoader
from .dataset_syntax import DatasetSyntax
from .preprocessor import RowPreprocessor
from .register import DATASET_MAPPING, DatasetMeta, SubsetDataset

logger = get_logger()


def get_dataset_num_samples(
    dataset_id: str,
    split: str = "train",
    subset: Optional[str] = None,
    source: str = "huggingface",
    fallback_stream_count: bool = True,
    count_all: bool = False,
    stream_count_limit: int = 100000,
) -> Optional[int]:
    """
    Get the number of samples in a HuggingFace/ModelScope dataset.
    
    IMPORTANT: Not all datasets have metadata with num_examples!
    This function tries multiple methods:
    1. First, try to get from dataset metadata (fast, but not always available)
    2. If metadata not available and fallback_stream_count=True, stream-count samples
    3. Returns None only if all methods fail
    
    Args:
        dataset_id: Dataset identifier (e.g., 'tatsu-lab/alpaca')
        split: Dataset split to query (default: 'train')
        subset: Optional subset/config name
        source: 'huggingface' or 'modelscope'
        fallback_stream_count: If True, count by streaming when metadata unavailable
        count_all: If True, count ALL samples without limit (may take hours for huge datasets)
        stream_count_limit: Max samples to count when streaming (ignored if count_all=True)
        
    Returns:
        Number of samples if available, None if cannot determine
        
    Note:
        - When count_all=True: Always returns exact count (may take long time)
        - When count_all=False and metadata unavailable: returns minimum estimate
    """
    num_samples = None
    metadata_available = False
    
    # ========== METHOD 1: Try to get from metadata (fast) ==========
    if source == "huggingface":
        try:
            from datasets import load_dataset_builder
            
            builder_kwargs = {"path": dataset_id, "trust_remote_code": True}
            if subset:
                builder_kwargs["name"] = subset
            
            builder = load_dataset_builder(**builder_kwargs)
            info = builder.info
            
            if info and info.splits and split in info.splits:
                num_examples = info.splits[split].num_examples
                if num_examples is not None and num_examples > 0:
                    logger.info(f"[Metadata] Dataset '{dataset_id}' has {num_examples:,} samples in '{split}' split")
                    return num_examples
                else:
                    logger.debug(f"Dataset '{dataset_id}' metadata exists but num_examples is None/0")
            else:
                logger.debug(f"Dataset '{dataset_id}' has no split info in metadata")
                
        except Exception as e:
            logger.debug(f"Could not get HuggingFace metadata for {dataset_id}: {e}")
    
    elif source == "modelscope":
        try:
            from modelscope.msdatasets import MsDataset
            
            # ModelScope: Try to get dataset info
            # Note: ModelScope API may vary, this is best-effort
            try:
                from modelscope.hub.api import HubApi
                api = HubApi()
                dataset_info = api.get_dataset(dataset_id)
                if hasattr(dataset_info, 'num_samples') and dataset_info.num_samples:
                    logger.info(f"[Metadata] ModelScope '{dataset_id}' has {dataset_info.num_samples:,} samples")
                    return dataset_info.num_samples
            except Exception:
                pass
                
        except ImportError:
            logger.debug("ModelScope not installed")
        except Exception as e:
            logger.debug(f"Could not get ModelScope metadata for {dataset_id}: {e}")
    
    # ========== METHOD 2: Fallback - stream and count ==========
    if fallback_stream_count and num_samples is None:
        effective_limit = None if count_all else stream_count_limit
        limit_msg = "no limit - counting ALL" if count_all else f"limit: {stream_count_limit:,}"
        logger.info(f"Metadata not available for '{dataset_id}', counting by streaming ({limit_msg})...")
        
        try:
            count = 0
            last_log = 0
            
            if source == "huggingface":
                from datasets import load_dataset as hf_load
                
                load_kwargs = {
                    "path": dataset_id,
                    "split": split,
                    "streaming": True,
                    "trust_remote_code": True,
                }
                if subset:
                    load_kwargs["name"] = subset
                
                ds = hf_load(**load_kwargs)
                for _ in ds:
                    count += 1
                    # Progress logging for large counts
                    if count_all and count - last_log >= 1_000_000:
                        logger.info(f"  Counting... {count:,} samples so far")
                        last_log = count
                    if not count_all and count >= stream_count_limit:
                        logger.warning(
                            f"Dataset '{dataset_id}' has >= {stream_count_limit:,} samples. "
                            f"Stopped at limit. Set count_all=True to count ALL samples."
                        )
                        break
                        
            elif source == "modelscope":
                from modelscope.msdatasets import MsDataset
                
                load_kwargs = {"dataset_name": dataset_id, "split": split}
                if subset:
                    load_kwargs["subset_name"] = subset
                
                ds = MsDataset.load(**load_kwargs)
                for _ in ds:
                    count += 1
                    if count_all and count - last_log >= 1_000_000:
                        logger.info(f"  Counting... {count:,} samples so far")
                        last_log = count
                    if not count_all and count >= stream_count_limit:
                        logger.warning(
                            f"ModelScope '{dataset_id}' has >= {stream_count_limit:,} samples. "
                            f"Stopped at limit. Set count_all=True to count ALL samples."
                        )
                        break
            
            if count > 0:
                is_exact = count_all or count < stream_count_limit
                if is_exact:
                    logger.info(f"✓ [Stream-count] Dataset '{dataset_id}' has exactly {count:,} samples")
                else:
                    logger.info(f"⚠ [Stream-count] Dataset '{dataset_id}' has >= {count:,} samples (limit reached)")
                return count
                
        except Exception as e:
            logger.warning(f"Failed to stream-count dataset {dataset_id}: {e}")
    
    return None


def count_local_dataset_samples(
    dataset_path: str,
    max_count: Optional[int] = None,
    show_progress: bool = True,
) -> int:
    """
    Count samples in a local dataset file by streaming through it.
    
    For massive datasets (100TB+, billions of rows), this provides an accurate 
    count without loading the full dataset into memory.
    
    Supports: JSONL, JSON, CSV, TXT, Parquet
    
    Args:
        dataset_path: Path to local dataset file
        max_count: Stop counting after this many samples (for estimation)
        show_progress: Log progress for large files
        
    Returns:
        Number of samples counted (0 if error)
        
    Example:
        >>> # Count all samples in a 100TB JSONL file
        >>> total = count_local_dataset_samples('/data/massive_dataset.jsonl')
        >>> print(f"Total samples: {total:,}")  # e.g., 50,000,000,000
    """
    if not os.path.exists(dataset_path):
        logger.error(f"File not found: {dataset_path}")
        return 0
    
    count = 0
    ext = os.path.splitext(dataset_path)[1].lower()
    file_size = os.path.getsize(dataset_path)
    file_size_gb = file_size / (1024 ** 3)
    
    logger.info(f"Counting samples in {dataset_path} ({file_size_gb:.2f} GB)...")
    
    try:
        # ========== JSONL - Line counting (fastest) ==========
        if ext == '.jsonl':
            with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.strip() and line.strip() != '':
                        count += 1
                        if max_count and count >= max_count:
                            break
                    # Progress logging for large files
                    if show_progress and count % 10_000_000 == 0:
                        logger.info(f"  Counted {count:,} samples so far...")
        
        # ========== JSON - Must parse (limited to 2GB files) ==========
        elif ext == '.json':
            import json
            if file_size_gb > 2.0:
                logger.warning(f"JSON file is {file_size_gb:.1f}GB - may run out of memory. Use JSONL for large files.")
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                count = len(data) if isinstance(data, list) else 1
        
        # ========== CSV - Row counting ==========
        elif ext == '.csv':
            with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Skip header
                next(f, None)
                for line in f:
                    if line.strip():
                        count += 1
                        if max_count and count >= max_count:
                            break
                    if show_progress and count % 10_000_000 == 0:
                        logger.info(f"  Counted {count:,} samples so far...")
        
        # ========== TXT - Line counting ==========
        elif ext == '.txt':
            with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.strip():
                        count += 1
                        if max_count and count >= max_count:
                            break
                    if show_progress and count % 10_000_000 == 0:
                        logger.info(f"  Counted {count:,} samples so far...")
        
        # ========== Parquet - Use pyarrow for row count ==========
        elif ext == '.parquet':
            try:
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(dataset_path)
                count = parquet_file.metadata.num_rows
                logger.info(f"  Parquet metadata: {count:,} rows")
            except ImportError:
                logger.warning("pyarrow not installed, cannot count Parquet rows")
            except Exception as e:
                logger.warning(f"Error reading Parquet metadata: {e}")
        
        # ========== Other formats - Use HuggingFace datasets ==========
        else:
            try:
                from datasets import load_dataset as hf_load
                ds = hf_load(ext.lstrip('.'), data_files=dataset_path, split='train', streaming=True)
                for _ in ds:
                    count += 1
                    if max_count and count >= max_count:
                        break
                    if show_progress and count % 1_000_000 == 0:
                        logger.info(f"  Counted {count:,} samples so far...")
            except Exception as e:
                logger.warning(f"Could not count using HuggingFace datasets: {e}")
        
        if count > 0:
            logger.info(f"✓ Total samples in {os.path.basename(dataset_path)}: {count:,}")
        
    except Exception as e:
        logger.error(f"Error counting samples in {dataset_path}: {e}")
        
    return count


def count_dataset_samples(
    dataset_path: str,
    streaming: bool = True,
    max_count: Optional[int] = None,
) -> int:
    """
    DEPRECATED: Use count_local_dataset_samples() instead.
    
    This function is kept for backward compatibility.
    """
    return count_local_dataset_samples(dataset_path, max_count=max_count)


def get_multiple_datasets_sample_info(
    dataset_paths: List[str],
    count_all: bool = False,
    stream_count_limit: int = 100000,
) -> Dict[str, Any]:
    """
    Get sample count information for multiple datasets.
    
    Handles the case where some datasets have known counts and others don't.
    
    Args:
        dataset_paths: List of dataset paths (local paths or HF::id/MS::id format)
        count_all: If True, count ALL samples (may take hours)
        stream_count_limit: Limit for streaming count per dataset
        
    Returns:
        Dictionary with:
        - 'datasets': List of dicts with path, count, count_type ('exact', 'estimate', 'unknown')
        - 'total_known': Sum of all known counts
        - 'all_counts_known': True if all datasets have exact counts
        - 'has_unknown': True if any dataset has unknown count
        - 'recommendation': Suggested training strategy
        
    Example:
        >>> info = get_multiple_datasets_sample_info([
        ...     '/data/local.jsonl',
        ...     'HF::tatsu-lab/alpaca',
        ...     'HF::unknown-dataset/no-metadata'
        ... ])
        >>> if info['has_unknown']:
        ...     print("Some datasets have unknown counts")
        ...     print(f"Recommendation: {info['recommendation']}")
    """
    results = {
        'datasets': [],
        'total_known': 0,
        'all_counts_known': True,
        'has_unknown': False,
        'recommendation': None,
    }
    
    for path in dataset_paths:
        ds_info = {'path': path, 'count': None, 'count_type': 'unknown'}
        
        # Parse path to determine source
        if path.upper().startswith('HF::'):
            # HuggingFace dataset
            dataset_id = path[4:]
            count = get_dataset_num_samples(
                dataset_id, 
                source='huggingface',
                count_all=count_all,
                stream_count_limit=stream_count_limit
            )
            if count is not None:
                ds_info['count'] = count
                ds_info['count_type'] = 'exact' if count_all else 'estimate'
                results['total_known'] += count
            else:
                ds_info['count_type'] = 'unknown'
                results['all_counts_known'] = False
                results['has_unknown'] = True
                
        elif path.upper().startswith('MS::'):
            # ModelScope dataset
            dataset_id = path[4:]
            count = get_dataset_num_samples(
                dataset_id,
                source='modelscope',
                count_all=count_all,
                stream_count_limit=stream_count_limit
            )
            if count is not None:
                ds_info['count'] = count
                ds_info['count_type'] = 'exact' if count_all else 'estimate'
                results['total_known'] += count
            else:
                ds_info['count_type'] = 'unknown'
                results['all_counts_known'] = False
                results['has_unknown'] = True
                
        else:
            # Local file - can always count exactly
            if os.path.exists(path):
                count = count_local_dataset_samples(path)
                ds_info['count'] = count
                ds_info['count_type'] = 'exact'
                results['total_known'] += count
            else:
                ds_info['count_type'] = 'unknown'
                results['all_counts_known'] = False
                results['has_unknown'] = True
        
        results['datasets'].append(ds_info)
    
    # Generate recommendation
    if results['all_counts_known']:
        results['recommendation'] = (
            f"All {len(dataset_paths)} datasets have known counts. "
            f"Total: {results['total_known']:,} samples. "
            f"Can calculate exact max_steps."
        )
    elif results['has_unknown']:
        unknown_count = sum(1 for d in results['datasets'] if d['count_type'] == 'unknown')
        results['recommendation'] = (
            f"{unknown_count}/{len(dataset_paths)} datasets have unknown sample counts. "
            f"For streaming mode, set --max_steps manually or use count_all=True to get exact count first."
        )
    
    logger.info(f"Multiple dataset sample info: {results['recommendation']}")
    return results


def calculate_max_steps_for_streaming(
    num_samples: int,
    num_epochs: float,
    batch_size: int,
    gradient_accumulation_steps: int = 1,
    world_size: int = 1,
) -> int:
    """
    Calculate max_steps for streaming training to cover all samples.
    
    This allows training on ALL samples without manually setting max_steps.
    
    Args:
        num_samples: Total number of samples in dataset
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        world_size: Number of distributed processes
        
    Returns:
        max_steps value to train on all samples for specified epochs
        
    Example:
        >>> # 1 billion samples, 1 epoch, batch 8, 8 GPUs
        >>> max_steps = calculate_max_steps_for_streaming(
        ...     num_samples=1_000_000_000,
        ...     num_epochs=1.0,
        ...     batch_size=8,
        ...     gradient_accumulation_steps=4,
        ...     world_size=8
        ... )
        >>> print(f"max_steps: {max_steps:,}")  # 3,906,250 steps
    """
    total_batch_size = batch_size * gradient_accumulation_steps * world_size
    steps_per_epoch = num_samples // total_batch_size
    max_steps = int(steps_per_epoch * num_epochs)
    
    logger.info(
        f"Calculated max_steps={max_steps:,} for {num_samples:,} samples, "
        f"{num_epochs} epochs, batch_size={batch_size}, "
        f"grad_accum={gradient_accumulation_steps}, world_size={world_size}"
    )
    
    return max(max_steps, 1)


class DatasetLoader(BaseDatasetLoader):

    def __init__(
        self,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        streaming: bool = False,
        hub_token: Optional[str] = None,
        strict: bool = False,
        download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
        columns: Optional[Dict[str, str]] = None,
        remove_unused_columns: bool = True,
    ):
        self.num_proc = num_proc
        self.load_from_cache_file = load_from_cache_file
        self.streaming = streaming
        self.hub_token = hub_token
        self.strict = strict
        self.download_mode = download_mode
        self.columns = columns
        self.remove_unused_columns = remove_unused_columns

    def _load_dataset_path(
        self,
        dataset_path: str,
        dataset_meta: DatasetMeta,
    ) -> HfDataset:
        ext = os.path.splitext(dataset_path)[1].lstrip('.')
        file_type = {'jsonl': 'json', 'txt': 'text'}.get(ext) or ext
        kwargs = {'split': 'train', 'streaming': self.streaming, 'num_proc': self.num_proc}
        if file_type == 'csv':
            kwargs['na_filter'] = False
        with safe_ddp_context(None, True):
            kwargs['cache_dir'] = os.path.join(get_cache_dir(), 'datasets')
            dataset = hf_load_dataset(file_type, data_files=dataset_path, **kwargs)
        if self.columns:
            dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns)
        dataset = dataset_meta.preprocess_func(
            dataset, num_proc=self.num_proc, load_from_cache_file=self.load_from_cache_file, strict=self.strict)
        if self.remove_unused_columns:
            dataset = RowPreprocessor.remove_useless_columns(dataset)
        return dataset

    def _load_repo_dataset(
        self,
        dataset_id: str,
        subset: SubsetDataset,
        *,
        use_hf: Optional[bool] = None,
        revision: Optional[str] = None,
    ) -> HfDataset:
        datasets = []
        if os.path.isdir(dataset_id):
            retry = 1
            load_context = nullcontext
            use_hf = True
            dataset_str = f'Use local folder, dataset_dir: {dataset_id}'
            # Local directory - validate as local source
            validate_dataset_source('local', dataset_id)
            # The dataset downloaded from modelscope will have an additional dataset_infos.json file.
            with safe_ddp_context('dataset_infos_rename'):
                dataset_infos_path = os.path.join(dataset_id, 'dataset_infos.json')
                if os.path.isfile(dataset_infos_path):
                    os.rename(dataset_infos_path, f'{dataset_infos_path}_bak')
        elif dataset_id.startswith('/'):
            raise ValueError(f'The local path does not exist, dataset_id: `{dataset_id}`. '
                             f'os.path.exists(dataset_id): {os.path.exists(dataset_id)}')
        else:
            retry = 3
            load_context = partial(safe_ddp_context, hash_id=dataset_id, use_barrier=True)
            dataset_str_f = 'Downloading the dataset from {hub}, dataset_id: {dataset_id}'
            if use_hf:
                # Validate HuggingFace source before attempting to load
                validate_dataset_source('huggingface', dataset_id)
                dataset_str = dataset_str_f.format(hub='HuggingFace', dataset_id=dataset_id)
            else:
                # Validate ModelScope source before attempting to load
                validate_dataset_source('modelscope', dataset_id)
                dataset_str = dataset_str_f.format(hub='US Inc', dataset_id=dataset_id)
        logger.info_debug(dataset_str)
        hub = get_hub(use_hf)
        for split in subset.split:
            i = 1
            with load_context():
                while True:
                    try:
                        dataset = hub.load_dataset(
                            dataset_id,
                            subset.subset,
                            split,
                            streaming=self.streaming,
                            revision=revision,
                            download_mode=self.download_mode,
                            hub_token=self.hub_token,
                            num_proc=self.num_proc)
                    except Exception as e:
                        if i == retry:
                            raise
                        i += 1
                        logger.error(f'Dataset {dataset_id} load failed: subset_name={subset.subset},'
                                     f'split={split} with error: {e}')
                    else:
                        break
            if hasattr(dataset, '_hf_ds'):
                dataset = dataset._hf_ds
                if self.streaming and isinstance(dataset, HfDataset):
                    dataset = dataset.to_iterable_dataset()
            if self.columns:
                dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns)
            dataset = subset.preprocess_func(
                dataset, num_proc=self.num_proc, load_from_cache_file=self.load_from_cache_file, strict=self.strict)
            if self.remove_unused_columns:
                dataset = RowPreprocessor.remove_useless_columns(dataset)
            datasets.append(dataset)
        return self.concat_datasets(datasets)

    @staticmethod
    def _select_subsets(subsets: List[str], dataset_meta: DatasetMeta) -> List[SubsetDataset]:
        subset_mapping = {subset.name: subset for subset in dataset_meta.subsets}
        subset_names = list(subset_mapping.keys())
        if not subsets:
            if len(subset_names) <= 1:
                subsets = subset_names
            elif 'default' in subset_names:
                subsets = ['default']
            else:
                raise ValueError(f'Please provide subsets. available subsets: {subset_names}')
        elif len(subsets) == 1 and subsets[0] == 'all' and 'all' not in subset_names:
            subsets = [subset_name for subset_name in subset_names if not subset_mapping[subset_name].is_weak_subset]

        subsets = [
            subset_mapping[subset_name] if subset_name in subset_mapping else SubsetDataset(subset=subset_name)
            for subset_name in subsets
        ]
        return [subset.set_default(dataset_meta) for subset in subsets]

    def load(
        self,
        dataset_syntax: Optional[DatasetSyntax] = None,
        dataset_meta: Optional[DatasetMeta] = None,
        *,
        use_hf: Optional[bool] = None,
    ) -> HfDataset:
        if dataset_syntax.dataset_type == 'path':
            dataset = self._load_dataset_path(
                dataset_syntax.dataset,
                dataset_meta=dataset_meta,
            )
        else:
            subsets: List[SubsetDataset] = self._select_subsets(dataset_syntax.subsets, dataset_meta)
            revision = dataset_meta.hf_revision if use_hf else dataset_meta.ms_revision
            datasets = []
            for subset in subsets:
                dataset = self._load_repo_dataset(
                    dataset_syntax.dataset,
                    subset,
                    use_hf=use_hf,
                    revision=revision,
                )
                datasets.append(dataset)
            dataset = self.concat_datasets(datasets)
        return dataset


def init_self_cognition_preprocessor(
    dataset_meta: Optional[DatasetMeta],
    model_name: Optional[Union[Tuple[str, str], List[str]]] = None,
    model_author: Optional[Union[Tuple[str, str], List[str]]] = None,
) -> None:
    from .dataset.llm import SelfCognitionPreprocessor
    if dataset_meta is None or model_name is None and model_author is None:
        return
    kwargs = {}
    # zh, en
    for key in ['name', 'author']:
        val = locals()[f'model_{key}']
        if isinstance(val, str):
            val = [val]
        if val is not None and val[0] is not None and (len(val) == 1 or val[1] is None):
            val = (val[0], val[0])
        kwargs[key] = val

    preprocess_funcs = [dataset_meta.preprocess_func]
    preprocess_funcs += [subset.preprocess_func for subset in dataset_meta.subsets if isinstance(subset, SubsetDataset)]
    for preprocess_func in preprocess_funcs:
        if isinstance(preprocess_func, SelfCognitionPreprocessor):
            preprocess_func.set_name_author(**kwargs)
    logger.info_once(f"SelfCognitionPreprocessor has been successfully configured with name: {kwargs['name']}, "
                     f"author: {kwargs['author']}.")


def load_dataset(
    datasets: Union[List[str], str],
    *,
    split_dataset_ratio: float = 0.,
    seed: Union[int, np.random.RandomState, None] = 42,
    num_proc: int = 1,
    load_from_cache_file: bool = True,
    shuffle: bool = False,
    streaming: bool = False,
    interleave_prob: Optional[List[float]] = None,
    stopping_strategy: Literal['first_exhausted', 'all_exhausted'] = 'first_exhausted',
    shuffle_buffer_size: int = 1000,
    use_hf: Optional[bool] = None,
    hub_token: Optional[str] = None,
    strict: bool = False,
    download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
    columns: Optional[Dict[str, str]] = None,  # columns_mapping
    remove_unused_columns: bool = True,
    # self-cognition
    model_name: Optional[Union[Tuple[str, str], List[str]]] = None,  # zh, en
    model_author: Optional[Union[Tuple[str, str], List[str]]] = None,
) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
    """Load and preprocess datasets.

    This function provides a unified interface to load datasets from various sources (HuggingFace,
    US Inc, or local paths), with support for splitting, shuffling, streaming, and interleaving
    multiple datasets. It also handles self-cognition dataset preprocessing for model training.

    Args:
        datasets: Single dataset name or list of dataset names to load. Can use special syntax
            for advanced configurations (e.g., 'dataset_name#1000' for sampling).
        split_dataset_ratio: Ratio for splitting dataset into train/validation sets.
            Value between 0 and 1. If 0, no validation split is created. Default: 0.
        seed: Random seed for reproducibility. Can be an integer or numpy RandomState object.
            If None, results will be non-deterministic. Default: 42.
        num_proc: Number of processes to use for dataset preprocessing. Set to None for
            streaming mode. Default: 1.
        load_from_cache_file: Whether to load preprocessed data from cache if available.
            Default: True.
        shuffle: Whether to shuffle the dataset(s) after loading. Default: False.
        streaming: Enable streaming mode for large datasets that don't fit in memory.
            When True, num_proc is automatically set to None. Default: False.
        interleave_prob: Probability weights for interleaving multiple datasets. Must have
            same length as datasets list. If None, datasets are concatenated instead. Default: None.
        stopping_strategy: Strategy when interleaving datasets of different lengths:
            - 'first_exhausted': Stop when shortest dataset is exhausted
            - 'all_exhausted': Continue until all datasets are exhausted
            Default: 'first_exhausted'.
        shuffle_buffer_size: Buffer size for shuffling in streaming mode. Larger values
            provide better randomization but use more memory. Default: 1000.
        use_hf: Force using HuggingFace Hub (True) or US Inc (False). If None,
            it is controlled by the environment variable `USE_HF`, which defaults to '0'.
            Default: None.
        hub_token: Authentication token for accessing private datasets on the hub. Default: None.
        strict: If True, raise exceptions when encountering malformed data rows.
            If False, skip invalid rows with warnings. Default: False.
        download_mode: How to handle existing cached datasets:
            - 'reuse_dataset_if_exists': Use cached version if available
            - 'force_redownload': Always download fresh copy
            Default: 'reuse_dataset_if_exists'.
        columns: Manual column name mapping for datasets. Dictionary mapping source column
            names to target column names (e.g., {'text': 'content'}). Default: None.
        remove_unused_columns: Whether to remove columns not used in preprocessing.
            Helps reduce memory usage. Default: True.
        model_name: Model name for self-cognition task preprocessing. Can be a tuple of
            (Chinese_name, English_name) or list of names. Default: None.
        model_author: Model author for self-cognition task preprocessing. Can be a tuple of
            (Chinese_author, English_author) or list of authors. Default: None.

    Returns:
        A tuple of (train_dataset, val_dataset):
            - train_dataset: The training dataset
            - val_dataset: The validation dataset if split_dataset_ratio > 0, otherwise None

    Examples:
        >>> # Load single dataset
        >>> train_ds, val_ds = load_dataset('tatsu-lab/alpaca', split_dataset_ratio=0.1)

        >>> # Load multiple datasets
        >>> train_ds, _ = load_dataset(
        ...     ['tatsu-lab/alpaca#500', 'usf_bios/self-cognition#500'],
        ...     model_name=('My Model', 'MyModel'),
        ...     model_author=('Author', 'Author')
        ... )
    """
    init_self_cognition_preprocessor(DATASET_MAPPING.get('self-cognition'), model_name, model_author)
    if isinstance(datasets, str):
        datasets = [datasets]
    if not isinstance(seed, np.random.RandomState):
        seed = np.random.RandomState(seed)
    if streaming:
        num_proc = None
    train_datasets = []
    val_datasets = []
    loader = DatasetLoader(
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        streaming=streaming,
        hub_token=hub_token,
        strict=strict,
        download_mode=download_mode,
        columns=columns,  # columns_mapping
        remove_unused_columns=remove_unused_columns,
    )

    use_hf_default = use_hf
    if use_hf_default is None:
        use_hf_default = True if use_hf_hub() else False
    for dataset in datasets:
        dataset_syntax = DatasetSyntax.parse(dataset)
        use_hf = dataset_syntax.use_hf or use_hf_default
        # compat dataset_name
        if dataset_syntax.dataset in DATASET_MAPPING:
            dataset_meta = DATASET_MAPPING[dataset_syntax.dataset]
            if dataset_syntax.use_hf is None and dataset_meta.dataset_path is not None:
                dataset_syntax.dataset = dataset_meta.dataset_path
                dataset_syntax.dataset_type = 'path'
            else:
                dataset_syntax.dataset = dataset_meta.hf_dataset_id if use_hf else dataset_meta.ms_dataset_id
        else:
            dataset_meta = dataset_syntax.get_dataset_meta(use_hf)
        train_dataset = loader.load(dataset_syntax, dataset_meta, use_hf=use_hf)
        train_dataset, val_dataset = loader.post_process(
            train_dataset,
            dataset_sample=dataset_syntax.dataset_sample,
            split_dataset_ratio=split_dataset_ratio,
            streaming=streaming,
            shuffle=shuffle,
            random_state=seed,
        )
        if train_dataset is not None:
            train_datasets.append(train_dataset)
        if val_dataset is not None:
            val_datasets.append(val_dataset)

    if interleave_prob is None:
        train_datasets = loader.concat_datasets(train_datasets)
        val_datasets = loader.concat_datasets(val_datasets)
    else:
        train_datasets = loader.interleave_datasets(
            train_datasets, interleave_prob, seed=get_seed(seed), stopping_strategy=stopping_strategy)
        val_datasets = loader.interleave_datasets(
            val_datasets, interleave_prob, seed=get_seed(seed), stopping_strategy=stopping_strategy)

    if shuffle:
        if train_datasets:
            train_datasets = loader.shuffle_dataset(
                train_datasets, seed=get_seed(seed), buffer_size=shuffle_buffer_size)
        if val_datasets:
            val_datasets = loader.shuffle_dataset(val_datasets, seed=get_seed(seed), buffer_size=shuffle_buffer_size)
    return train_datasets, val_datasets
