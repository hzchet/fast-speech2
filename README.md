# fast-speech2
Implementation of the TTS model based on the FastSpeech2.

# Reproducing results
To reproduce training of the final model, follow the steps:

1. Specify the `GPUS` (gpu indices that will be used during trained), `SAVE_DIR` (directory where all the logs & checkpoints will be stored), `DATA_DIR (directory that will store the training data)`, `NOTEBOOK_DIR` (directory that contains your notebooks, for debugging purposes) in the `Makefile`. Set up `WANDB_API_KEY` variable in the Dockerfile to log the training process.

2. Build and run `Docker` container
```shell
make build && make run
```

3. Install alignments, LJSpeech-1.1 dataset and pre-calculated mel-spectrograms by running shell commands (in the given order)
```shell
cd data
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar.bz2 
```
```shell
gdown https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx
gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
mkdir -p waveglow/pretrained_model/
mv waveglow_256channels_ljs_v2.pt waveglow/pretrained_model/waveglow_256channels.pt
```
```shell
gdown https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j
tar -xvf mel.tar.gz
```
```shell
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip
```

4. Pre-calculate the energy and pitch of the given wavs by running
```shell
python3 precalculate_features.py -c configs/feature_extractor.json
```

5. Run the pipeline described in the `configs/fastspeech2.json`
```shell
python3 train.py -c configs/fastspeech2.json
```

# Running inference
1. In order to run an inference on the pre-trained model, you should first download its weights by running
```shell
python3 install_weights.py
```
2. Copy the config file into the same directory as the model weights
```shell
cp configs/fastspeech2.json saved/models/final/
```
3. Write your desired prompt texts in the file in the following format:
```shell
text_1
text_2
...
text_n
```
4. Run
```shell
python3 test.py -r saved/models/final/weights.pth -t <PATH_TO_PROMPTs_FILE> -o <OUTPUT_DIR>
```
5. Listen to generated audios, that are now stored in the `<OUTPUT_DIR>` folder.
