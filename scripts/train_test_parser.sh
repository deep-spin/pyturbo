cd ..

HOME=/Users/afm

python3 parser/turbo_parser.py \
        --train \
        --unlabeled \
        --training_path $HOME/projects/TurboParser/data/pt-ud-normalized/pt-ud-normalized_train.conll \
        --training_epochs 1 \
        --model_path model.pkl

python3 parser/turbo_parser.py \
        --test \
        --evaluate \
        --test_path $HOME/projects/TurboParser/data/pt-ud-normalized/pt-ud-normalized_test.conll \
        --model_path model.pkl \
        --output_path out.conll

perl scripts/eval.pl -b -q -g $HOME/projects/TurboParser/data/pt-ud-normalized/pt-ud-normalized_test.conll -s out.conll | tail -5
