# en-ch-NMT
a Pytorch implementation of neural machine translation system from english (chinese) to chinese (english) based on 30m parallel data.

this is a neural machine translation system from english to chinese and vice versa. 

The training set contians 30m english parallel corpora from chinese to english.

As we use Google's sentencepiece encoding methods for both en and ch, the shared vocabulary enables us to learn a single model to translate either english or chinese.

The model is a 8 layer 512 transformer enhanced with LSTM.

I use a webservice to make the model output more instant visible.

------------------------
Usage:

1）pip3 install -r requirements.txt

2）create a directory model, and download the model from  https://drive.google.com/file/d/1I8P2t4UxJSkP2kfyNA7LJbrEexoSl0gX/view?usp=sharing

3）python3 interface.py

and then open your browser with http://localhost:4000/

---------------------------
I also build a public website: https://bingning.wang/translation/ (due to the mvps memory limits, the model is a 512-5 layer transformer).


some snapshots of our model.

![Image text](https://raw.githubusercontent.com/benywon/en-ch-NMT/master/WX20181216-111022%402x.png)
![Image text](https://raw.githubusercontent.com/benywon/en-ch-NMT/master/WX20181216-111101%402x.png)
