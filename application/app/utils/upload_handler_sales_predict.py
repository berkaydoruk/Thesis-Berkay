import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'

def save_uploaded_files(file1, file2):
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    filename1 = secure_filename(file1.filename)
    filepath1 = os.path.join(UPLOAD_FOLDER, filename1)
    file1.save(filepath1)
    
    filename2 = secure_filename(file2.filename)
    filepath2 = os.path.join(UPLOAD_FOLDER, filename2)
    file2.save(filepath2)
    
    return filepath1, filepath2
