SHELL := /bin/bash

IMAGE_NAME := mlkit
VERSION := $(shell git describe --abbrev=0 --tags --always)

test:
	tox

install:
	python setup.py install

build:
	docker build -t $(IMAGE_NAME) .

.PHONY: test install build
