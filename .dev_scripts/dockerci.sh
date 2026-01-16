#!/bin/bash
HF_HOME_DIR_IN_CONTAINER=/hf_cache
CODE_DIR=$PWD
CODE_DIR_IN_CONTAINER=/usf-bios
USF_DEBUG=True
echo "$USER"
gpus='0,1 2,3'
cpu_sets='0-15 16-31'
cpu_sets_arr=($cpu_sets)
is_get_file_lock=false
CI_COMMAND=${CI_COMMAND:-bash .dev_scripts/ci_container_test.sh python tests/run.py --parallel 2 --run_config tests/run_config.yaml}
echo "ci command: $CI_COMMAND"
PR_CHANGED_FILES="${PR_CHANGED_FILES:-}"
echo "PR modified files: $PR_CHANGED_FILES"
PR_CHANGED_FILES=${PR_CHANGED_FILES//[ ]/#}
echo "PR_CHANGED_FILES: $PR_CHANGED_FILES"
idx=0
for gpu in $gpus
do
  exec {lock_fd}>"/tmp/gpu$gpu" || exit 1
  flock -n "$lock_fd" || { echo "WARN: gpu $gpu is in use!" >&2; idx=$((idx+1)); continue; }
  echo "get gpu lock $gpu"

  CONTAINER_NAME="usf-ci-$idx"
  let is_get_file_lock=true

  # pull image if there are update
  docker pull ${IMAGE_NAME}:${IMAGE_VERSION}
  if [ "$USF_DEBUG" == "True" ]; then
    echo 'debugging'
    docker run --rm --name $CONTAINER_NAME --shm-size=16gb \
              --cpuset-cpus=${cpu_sets_arr[$idx]} \
              --gpus='"'"device=$gpu"'"' \
              -v $CODE_DIR:$CODE_DIR_IN_CONTAINER \
              -v $HF_HOME:$HF_HOME_DIR_IN_CONTAINER \
              -v $HF_HOME/$idx:/root \
              -v /home/admin/pre-commit:/home/admin/pre-commit \
              -e CI_TEST=True \
              -e TEST_LEVEL=$TEST_LEVEL \
              -e HF_HOME=$HF_HOME_DIR_IN_CONTAINER \
              -e HF_ENDPOINT=$HF_ENDPOINT \
              -e USF_DEBUG=True \
              -e HUB_DATASET_ENDPOINT=$HUB_DATASET_ENDPOINT \
              -e TEST_ACCESS_TOKEN_CITEST=$TEST_ACCESS_TOKEN_CITEST \
              -e TEST_ACCESS_TOKEN_SDKDEV=$TEST_ACCESS_TOKEN_SDKDEV \
              -e TEST_LEVEL=$TEST_LEVEL \
              -e USF_ENVIRONMENT='ci' \
              -e TEST_UPLOAD_MS_TOKEN=$TEST_UPLOAD_MS_TOKEN \
              -e MODEL_TAG_URL=$MODEL_TAG_URL \
              -e HF_TOKEN=$HF_TOKEN \
	            -e PR_CHANGED_FILES=$PR_CHANGED_FILES \
              --workdir=$CODE_DIR_IN_CONTAINER \
              ${IMAGE_NAME}:${IMAGE_VERSION} \
              $CI_COMMAND
  else
    docker run --rm --name $CONTAINER_NAME --shm-size=16gb \
              --cpuset-cpus=${cpu_sets_arr[$idx]} \
              --gpus='"'"device=$gpu"'"' \
              -v $CODE_DIR:$CODE_DIR_IN_CONTAINER \
              -v $HF_HOME:$HF_HOME_DIR_IN_CONTAINER \
              -v $HF_HOME/$idx:/root \
              -v /home/admin/pre-commit:/home/admin/pre-commit \
              -e CI_TEST=True \
              -e TEST_LEVEL=$TEST_LEVEL \
              -e HF_HOME=$HF_HOME_DIR_IN_CONTAINER \
              -e HF_ENDPOINT=$HF_ENDPOINT \
              -e HUB_DATASET_ENDPOINT=$HUB_DATASET_ENDPOINT \
              -e TEST_ACCESS_TOKEN_CITEST=$TEST_ACCESS_TOKEN_CITEST \
              -e TEST_ACCESS_TOKEN_SDKDEV=$TEST_ACCESS_TOKEN_SDKDEV \
              -e TEST_LEVEL=$TEST_LEVEL \
              -e USF_ENVIRONMENT='ci' \
              -e TEST_UPLOAD_MS_TOKEN=$TEST_UPLOAD_MS_TOKEN \
              -e MODEL_TAG_URL=$MODEL_TAG_URL \
              -e HF_TOKEN=$HF_TOKEN \
	            -e PR_CHANGED_FILES=$PR_CHANGED_FILES \
              --workdir=$CODE_DIR_IN_CONTAINER \
              ${IMAGE_NAME}:${IMAGE_VERSION} \
              $CI_COMMAND
  fi
  if [ $? -ne 0 ]; then
    echo "Running test case failed, please check the log!"
    exit -1
  fi
  break
done
if [ "$is_get_file_lock" = false ] ; then
    echo 'No free GPU!'
    exit 1
fi
