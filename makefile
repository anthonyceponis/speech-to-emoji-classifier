setup: setup_website setup_model

setup_website:
	cd website && npm install

setup_model:
	cd model && \
	python3 -m venv env && \
	. env/bin/activate && \
	pip install -r requirements.txt && \
	python3 train.py

run:
	cd model && . env/bin/activate && python infer_server.py & # starts model inference server in bg
	cd website && npm run dev # starts website server in fg
