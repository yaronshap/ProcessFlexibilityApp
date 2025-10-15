# Troubleshooting Guide

## Common Errors and Solutions

### 1. "ModuleNotFoundError" or Import Errors
**Error**: `ModuleNotFoundError: No module named 'streamlit'`

**Solution**:
```bash
pip install -r requirements.txt
```

### 2. "FileNotFoundError" when running streamlit
**Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'app.py'`

**Solution**: Make sure you're in the correct directory:
```bash
# Check current directory
pwd  # or dir on Windows

# Navigate to the project directory
cd "path/to/your/project/directory"

# Then run the app
streamlit run app.py
```

### 3. "Port already in use" Error
**Error**: `Port 8501 is already in use`

**Solution**: 
- Close other Streamlit apps
- Or use a different port: `streamlit run app.py --server.port 8502`

### 4. "Permission denied" Error
**Error**: `PermissionError: [Errno 13] Permission denied`

**Solution**: 
- Run as administrator
- Check if the file is not read-only
- Make sure you have write permissions

### 5. "SyntaxError" in the code
**Error**: `SyntaxError: invalid syntax`

**Solution**: 
- Check Python version (requires Python 3.7+)
- Verify all imports are available
- Check for typos in the code

## Quick Fixes

### Install Dependencies
```bash
pip install streamlit networkx plotly numpy pandas
```

### Verify Installation
```bash
python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"
```

### Run with Debug Info
```bash
streamlit run app.py --logger.level debug
```

## Still Having Issues?

1. **Check Python version**: `python --version` (should be 3.7+)
2. **Check if packages are installed**: `pip list | findstr streamlit`
3. **Try running in a virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Contact
If you're still experiencing issues, please share:
1. The exact error message
2. Your Python version
3. Your operating system
4. The command you used to run the app
