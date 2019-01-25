#!make

include .env
export $(shell sed 's/=.*//' .env)

all: esc50

$(CS230_RAW_DATA)/esc50:
	mkdir -p $(CS230_RAW_DATA)/esc50

esc50: $(CS230_RAW_DATA)/esc50
	git clone https://github.com/karoldvl/ESC-50 $(CS230_RAW_DATA)/esc50

