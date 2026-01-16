# Copyright (c) US Inc. All rights reserved.
"""
Production-ready console UI for USF BIOS fine-tuning platform.
Provides beautiful, informative output for enterprise users.
"""
import os
import signal
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from usf_bios.version import __version__, __product_name__

# Global console instance
_console = Console() if RICH_AVAILABLE else None

# Training start time for duration calculation
_training_start_time = None

# Signal handler for graceful exit
_original_sigint_handler = None


def _get_model_display_name(model_path: str) -> str:
    """Get a clean display name for the model."""
    if not model_path:
        return "Unknown"
    
    # Check if it's a local path
    if os.path.exists(model_path):
        return os.path.basename(model_path)
    
    # Check if it's a HuggingFace/ModelScope model ID
    if '/' in model_path:
        return model_path.split('/')[-1]
    
    return model_path


def _get_model_source(model_path: str, use_hf: bool = True) -> str:
    """Determine the source of the model."""
    if not model_path:
        return "Unknown"
    
    if os.path.exists(model_path):
        return "Local"
    elif use_hf:
        return "HuggingFace"
    else:
        return "ModelScope"


def show_startup_banner(
    model: str = None,
    train_type: str = None,
    dataset_count: int = None,
    use_hf: bool = True,
    additional_info: Dict[str, Any] = None
) -> None:
    """Display professional startup banner."""
    if not RICH_AVAILABLE:
        print(f"\n{'='*60}")
        print(f"  {__product_name__} v{__version__}")
        print(f"  Fine-Tuning Platform")
        print(f"{'='*60}\n")
        return
    
    # Create header
    header = Text()
    header.append(f"{__product_name__}", style="bold cyan")
    header.append(f" v{__version__}", style="dim")
    
    # Create info table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value", style="bold white")
    
    if model:
        model_name = _get_model_display_name(model)
        model_source = _get_model_source(model, use_hf)
        table.add_row("Model", f"{model_name} ({model_source})")
    
    if train_type:
        train_type_display = train_type.upper() if train_type else "SFT"
        table.add_row("Training Type", train_type_display)
    
    if dataset_count is not None:
        table.add_row("Dataset", f"{dataset_count:,} samples")
    
    if additional_info:
        for key, value in additional_info.items():
            table.add_row(key, str(value))
    
    # Create panel
    panel = Panel(
        table,
        title=header,
        subtitle="Fine-Tuning Platform",
        border_style="cyan",
        box=box.DOUBLE,
        padding=(1, 2)
    )
    
    _console.print()
    _console.print(panel)
    _console.print()


def show_status(message: str, status: str = "info") -> None:
    """Show a status message with appropriate styling."""
    if not RICH_AVAILABLE:
        prefix = {"info": "[INFO]", "success": "[OK]", "warning": "[WARN]", "error": "[ERROR]"}.get(status, "[INFO]")
        print(f"{prefix} {message}")
        return
    
    styles = {
        "info": ("ℹ", "blue"),
        "success": ("✓", "green"),
        "warning": ("⚠", "yellow"),
        "error": ("✗", "red"),
        "loading": ("◌", "cyan"),
    }
    
    icon, color = styles.get(status, ("ℹ", "blue"))
    _console.print(f"[{color}]{icon}[/{color}] {message}")


def show_loading(message: str) -> None:
    """Show a loading status."""
    show_status(message, "loading")


def show_success(message: str) -> None:
    """Show a success status."""
    show_status(message, "success")


def show_error(message: str) -> None:
    """Show an error status."""
    show_status(message, "error")


def show_warning(message: str) -> None:
    """Show a warning status."""
    show_status(message, "warning")


@contextmanager
def loading_spinner(message: str):
    """Context manager for showing a loading spinner."""
    if not RICH_AVAILABLE:
        print(f"[...] {message}")
        yield
        print(f"[OK] {message} - Done")
        return
    
    with _console.status(f"[cyan]{message}...[/cyan]", spinner="dots"):
        yield
    show_success(f"{message} - Done")


def start_training_timer() -> None:
    """Start the training timer."""
    global _training_start_time
    _training_start_time = time.time()


def get_training_duration() -> str:
    """Get the training duration as a formatted string."""
    if _training_start_time is None:
        return "N/A"
    
    duration = time.time() - _training_start_time
    
    if duration < 60:
        return f"{duration:.1f}s"
    elif duration < 3600:
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        return f"{hours}h {minutes}m"


def show_training_complete(
    output_dir: str = None,
    final_loss: float = None,
    total_steps: int = None,
    additional_metrics: Dict[str, Any] = None
) -> None:
    """Display training completion summary."""
    duration = get_training_duration()
    
    if not RICH_AVAILABLE:
        print(f"\n{'='*60}")
        print("  Training Complete!")
        print(f"{'='*60}")
        if final_loss is not None:
            print(f"  Final Loss: {final_loss:.4f}")
        if duration != "N/A":
            print(f"  Duration: {duration}")
        if output_dir:
            print(f"  Output: {output_dir}")
        print(f"{'='*60}\n")
        return
    
    # Create summary table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold green")
    
    if final_loss is not None:
        table.add_row("Final Loss", f"{final_loss:.4f}")
    
    if total_steps is not None:
        table.add_row("Total Steps", f"{total_steps:,}")
    
    if duration != "N/A":
        table.add_row("Duration", duration)
    
    if additional_metrics:
        for key, value in additional_metrics.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))
    
    if output_dir:
        # Shorten path for display
        display_path = output_dir
        if len(display_path) > 50:
            display_path = "..." + display_path[-47:]
        table.add_row("Output", display_path)
    
    # Create panel
    panel = Panel(
        table,
        title="[bold green]✓ Training Complete[/bold green]",
        border_style="green",
        box=box.DOUBLE,
        padding=(1, 2)
    )
    
    _console.print()
    _console.print(panel)
    _console.print()


def show_training_failed(error_message: str = None) -> None:
    """Display training failure message."""
    if not RICH_AVAILABLE:
        print(f"\n{'='*60}")
        print("  Training Failed!")
        if error_message:
            print(f"  Error: {error_message}")
        print(f"{'='*60}\n")
        return
    
    content = Text()
    content.append("Training was interrupted or failed.\n\n", style="red")
    if error_message:
        content.append(f"Error: {error_message}", style="dim")
    
    panel = Panel(
        content,
        title="[bold red]✗ Training Failed[/bold red]",
        border_style="red",
        box=box.DOUBLE,
        padding=(1, 2)
    )
    
    _console.print()
    _console.print(panel)
    _console.print()


def setup_graceful_exit() -> None:
    """Setup graceful exit handling for Ctrl+C."""
    global _original_sigint_handler
    
    def signal_handler(signum, frame):
        if RICH_AVAILABLE:
            _console.print("\n[yellow]⚠ Interrupt received. Cleaning up...[/yellow]")
        else:
            print("\n[WARN] Interrupt received. Cleaning up...")
        
        # Call original handler to allow proper cleanup
        if _original_sigint_handler and callable(_original_sigint_handler):
            _original_sigint_handler(signum, frame)
        else:
            sys.exit(0)
    
    _original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)


def show_dataset_summary(
    total_samples: int,
    skipped_samples: int = 0,
    train_samples: int = None,
    val_samples: int = None
) -> None:
    """Display dataset preprocessing summary."""
    if not RICH_AVAILABLE:
        print(f"[INFO] Dataset: {total_samples:,} total samples")
        if skipped_samples > 0:
            print(f"[INFO] Skipped: {skipped_samples:,} samples")
        return
    
    remaining = total_samples - skipped_samples
    
    info_parts = [f"[bold]{remaining:,}[/bold] samples ready"]
    
    if skipped_samples > 0:
        info_parts.append(f"[dim]({skipped_samples:,} skipped)[/dim]")
    
    if train_samples is not None and val_samples is not None:
        info_parts.append(f"[dim](train: {train_samples:,}, val: {val_samples:,})[/dim]")
    
    _console.print(f"[green]✓[/green] Dataset: {' '.join(info_parts)}")


def show_model_info(
    model_name: str,
    model_type: str = None,
    parameters: str = None,
    source: str = None
) -> None:
    """Display model information."""
    if not RICH_AVAILABLE:
        print(f"[INFO] Model: {model_name}")
        return
    
    info_parts = [f"[bold]{model_name}[/bold]"]
    
    if model_type:
        info_parts.append(f"[dim]({model_type})[/dim]")
    
    if parameters:
        info_parts.append(f"[dim]- {parameters}[/dim]")
    
    if source:
        info_parts.append(f"[dim][{source}][/dim]")
    
    _console.print(f"[green]✓[/green] Model: {' '.join(info_parts)}")


# Export functions
__all__ = [
    'show_startup_banner',
    'show_status',
    'show_loading',
    'show_success',
    'show_error',
    'show_warning',
    'loading_spinner',
    'start_training_timer',
    'get_training_duration',
    'show_training_complete',
    'show_training_failed',
    'setup_graceful_exit',
    'show_dataset_summary',
    'show_model_info',
    'RICH_AVAILABLE',
]
