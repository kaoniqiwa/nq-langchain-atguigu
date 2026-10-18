SHELL := /bin/zsh

.PHONY: restart update check-shell

check-shell:
	@echo "========================================="
	@echo "1. 当前终端环境变量 SHELL: $$SHELL"
	@echo "2. Make 实际执行命令的 Shell (\$$0): $$0"
	@echo "3. Make 内部记录的 SHELL: $(SHELL)"
	@echo "========================================="
create:
	conda env create -f environment.yml
update:
	conda env update -n langchain -f environment.yml --prune
