INPUT=$1
GOLD=$2
MODEL=$3
PRUNER=$4

OUTPUT=$(basename $INPUT .conllu)-output.conllu
if [ -z "$PRUNER" ] 
then
    PRUNER_FLAG=''
else
    PRUNER_FLAG="--pruner_path $PRUNER"
fi
	 
python scripts/run-parser.py --test --test_path $INPUT --model_path $MODEL --output_path $OUTPUT --batch_size 128 --single_root $PRUNER_FLAG
python scripts/conll18_ud_eval.py $GOLD $OUTPUT -v
