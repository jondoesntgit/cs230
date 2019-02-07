#!make

include .env
export $(shell sed 's/=.*//' .env)

all: features

$(CS230_RAW_DATA)/esc50:
	mkdir -p $(CS230_RAW_DATA)
	git clone https://github.com/karoldvl/ESC-50 $(CS230_RAW_DATA)/esc50

features: | $(CS230_RAW_DATA)/esc50
	cd ./src/features && python extract_features.py $(CS230_RAW_DATA)/esc50/audio/2-118459-B-32.wav

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
