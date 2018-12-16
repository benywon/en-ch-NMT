# en-ch-NMT
a neural machine translation system from english (chinese) to chinese (english) based on 30m parallel data.

this is a neural machine translation system from english to chinese and vice versa. 

The training set contians 30m english parallel corpora from chinese to english.

As we use Google's sentencepiece encoding methods for both en and ch, the shared vocabulary enables us to learn a single model to translate either english or chinese.

The model is a 8 layer 512 transformer enhanced with LSTM. You can download the model at https://drive.google.com/file/d/1JMf3daDdMvi2GXH39EatcSmZws8cxFZW/view?usp=sharing

I use a webservice to make the model output more instant visible.

------------------------
Usage:

pip3 install -r requirements.txt
download the model from to the folder 'model'

python3 interface.py

and then open your browser with http://localhost:4000/

---------------------------

some snapshots of our model.

![Image text](https://raw.githubusercontent.com/benywon/en-ch-NMT/master/WX20181216-111022%402x.png)
![Image text](https://raw.githubusercontent.com/benywon/en-ch-NMT/master/WX20181216-111101%402x.png)















