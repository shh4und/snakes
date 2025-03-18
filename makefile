# Remove Python and Jupyter cache directories
clean:
	@echo "Cleaning Python and Jupyter cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +  2>/dev/null || true
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaning complete!"

# Display help information
help:
	@echo "Available targets:"
	@echo "  clean   - Remove all Python and Jupyter cache directories"
	@echo "  help    - Display this help message"

.PHONY: clean help