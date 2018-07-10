echo "builing envoronment..."
pip install tensorflow_gpu==1.4
pip install pandas sklearn nltk
python -c "import nltk; nltk.download()"
python builder.py
echo "now goto the att_rnn folder"
