TEX_NAME = main
BUILD_DIR = build
LATEXMK = latexmk
LATEXMK_FLAGS = -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD_DIR)

all:
	mkdir -p $(BUILD_DIR)
	$(LATEXMK) $(LATEXMK_FLAGS) $(TEX_NAME).tex

clean:
	-$(LATEXMK) -C -outdir=$(BUILD_DIR) $(TEX_NAME).tex
	rm -rf $(BUILD_DIR)
