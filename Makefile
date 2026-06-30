.PHONY: run train install

run:
	python3 -m streamlit run app/Home.py

train:
	python3 model/train.py

install:
	pip3 install -r requirements.txt
