TRAIN_TEST_SPLIT=navtest
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/dataset/maps"
export NAVSIM_EXP_ROOT="/path/to/DriveLaW-Act"
export NAVSIM_DEVKIT_ROOT="/path/to/DriveLaW-Act"
export OPENSCENE_DATA_ROOT="/path/to/dataset"
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
experiment_name=cv_agent