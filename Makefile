book:
	jupyter-book build ./
	if [ ! -d "docs" ]; then mkdir docs; fi
	if [ ! -f ".nojekyll" ]; then touch docs/.nojekyll; fi
	cp -r ./_build/html/* docs

clean-book:
	rm -rf docs/*
	rm -rf ./_build/*