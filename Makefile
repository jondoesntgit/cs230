#!make

include .env
export $(shell sed 's/=.*//' .env)

all: audioset

$(CS230_RAW_DATA)/esc50:
	mkdir -p $(CS230_RAW_DATA)
	git clone https://github.com/karoldvl/ESC-50 $(CS230_RAW_DATA)/esc50
	curl storage.googleapis.com/us_audioset/youtube_corpus/v1/features/features.tar.gz -o /tmp/features.tar.gz

vggish_params:
	mkdir -p $(dir $(VGGISH_MODEL_CHECKPOINT))
	mkdir -p $(dir $(EMBEDDING_PCA_PARAMETERS))
	curl -o $(VGGISH_MODEL_CHECKPOINT) https://storage.googleapis.com/audioset/vggish_model.ckpt
	curl -o $(EMBEDDING_PCA_PARAMETERS) https://storage.googleapis.com/audioset/vggish_pca_params.npz

audioset:
	ifndef AUDIOSET_PATH
	$(error AUDIOSET_PATH is not set)
	endif
	mkdir -p $(AUDIOSET_PATH)
	tar -xvzf /tmp/features.tar.gz -C $(AUDIOSET_PATH)
	cd $(AUDIOSET_PATH) && curl -O http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv

features: | $(CS230_RAW_DATA)/esc50
	cd ./src/features && python extract_features.py $(CS230_RAW_DATA)/esc50/audio/2-118459-B-32.wav

docker:
	docker build -t cs230 .

docker-ssh:
	docker run -i -t --entrypoint /bin/bash cs230

$(ESC50_SPLITS)/train/train_data.npy \
$(ESC50_SPLITS)/train/train_label.npy \
$(ESC50_SPLITS)/test/test_data.npy \
$(ESC50_SPLITS)/test/test_label.npy \
esc50-splits : | $(CS230_RAW_DATA)/esc50
	python ./src/data/esc50.py

esc50-piczak: $(ESC50_SPLITS)/train/train_data.npy \
			  $(ESC50_SPLITS)/train/train_label.npy \
			  $(ESC50_SPLITS)/test/test_data.npy \
			  $(ESC50_SPLITS)/test/test_label.npy
	python ./src/models/piczak.py --epochs=$(epochs)
