export ROOT=$(pwd)
export PYTHONPATH=$ROOT:$PYTHONPATH
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$ROOT/maps"
export NAVSIM_EXP_ROOT="$ROOT/exp"
export NAVSIM_DEVKIT_ROOT="$ROOT/"
export OPENSCENE_DATA_ROOT="$ROOT/dataset"

TRAIN_TEST_SPLIT=navtest
INFER_RESULT_PATH=$1
VISUALIZATION=$2
CACHE_PATH=exp/metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=covla_agent \
experiment_name=covla_agent \
metric_cache_path=$CACHE_PATH \
+infer_result_path=$INFER_RESULT_PATH \
+visualization=$VISUALIZATION

# bash scripts/evaluation/06_19_09_54_48.sh
