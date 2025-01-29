.PHONY: check_uv install requirements check test update help

check_uv: # install `uv` if not installed
	@if ! command -v uv > /dev/null 2>&1; then \
		echo "uv is not installed, installing now..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@uv self update

install: check_uv ## Install the virtual environment and pre-commit hooks
	@echo "📦 Creating virtual environment"
	@uv lock
	@uv sync --all-extras --frozen
	@echo "🛠️ Installing developer tools..."
	@if [ ! -d .git ]; then \
		echo "Initializing a new Git repository..."; \
		git init; \
		git add .; \
	fi
	@uv run pre-commit install

requirements: check_uv
	@echo "Exporting dependencies to requirements.txt..."
	@uv export --output-file requirements.txt

check: ## Run code quality tools
	@echo "🧹 Checking code: Running pre-commit"
	@uv run pre-commit run --all-files


test: ## Test the code with pytest
	@echo "✅ Testing code: Running pytest"
	@uv run pytest

notebook: ## Start a Jupyter notebook server
	@echo "📓 Starting Jupyter notebook server"
	@uv run jupyter lab

update: ## Update pre-commit hooks
	@echo "⚙️ Updating dependencies and pre-commit hooks"
	@uv lock --upgrade
	$(MAKE) requirements
	@uv run pre-commit autoupdate

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help