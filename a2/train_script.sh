TRAIN=/Users/tina/PycharmProjects/Training/
TEST=/Users/tina/PycharmProjects/Testing/
CELL_TYPE="gru"


## Variables TRAIN and TEST as before.
#export OMP_NUM_THREADS=4 # avoids a libgomp error on teach
## create an input and output vocabulary of only 100 words
#python3 a2_run.py vocab $TRAIN e vocab_tiny.e.gz --max-vocab 100
#python3 a2_run.py vocab $TRAIN f vocab_tiny.f.gz --max-vocab 100
## only use the proceedings of 4 meetings, 3 for training and 1 for dev
#python3 a2_run.py split $TRAIN train_tiny.txt.gz dev_tiny.txt.gz --limit 4
# use far fewer parameters in your model
python3 a2_run.py train $TRAIN \
vocab_tiny.e.gz vocab_tiny.f.gz \
train_tiny.txt.gz dev_tiny.txt.gz \
model.pt.gz \
--epochs 2 \
--word-embedding-size 51 \
--encoder-hidden-size 101 \
--batch-size 5 \
--cell-type lstm \
--beam-width 2 \
--with-attention