.PHONY: test

test: contourlet_sd_test

contourlet_sd_test:
	python -m unittest ./tests/contourlet_sd_test.py
