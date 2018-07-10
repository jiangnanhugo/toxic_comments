echo "builing envoronment..."
mkdir data
cd data && wget https://nlp.stanford.edu/data/glove.840B.300d.zip
cd data && unzip glove.840B.300d.zip
python3 -m pip install tensorflow_gpu==1.4
python3 -m pip install pandas sklearn nltk
python3 -c "import nltk; nltk.download()"
python3 builder.py
echo "now goto the att_rnn folder"
