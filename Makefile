dist: ## build source and wheel package
	python3 setup.py sdist bdist_wheel

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/

test: # run test with python
	pytest

release: dist ## package and upload a release
	twine upload dist/*	