# en-ch-NMT
a neural machine translation system from english (chinese) to chinese (english) based on 30m parallel data.

this is a neural machine translation system from english to chinese and vice versa. 

The training set contians 30m english parallel corpora from chinese to english.

As we use Google's sentencepiece encoding methods for both en and ch, the shared vocabulary enables us to learn a single model to translate either english or chinese.

The model is a 8 layer 512 transformer enhanced with LSTM. You can download the model at 

I use a webservice to make the model output more instant visible.

------------------------
Usage:

pip3 install -r requirements.txt
download the model from to the folder 'model'

python3 interface.py

and then open your browser with http://localhost:4000/

HAPPY TRANSLATION
---------------------------

some snapshots of our model.














