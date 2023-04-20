import filecmp
import os
import shutil
import tempfile

# return the itens in array that is not inexisting or empty
def validate_files(paths):
    valid_files = []
    for path in paths:
        if os.path.isfile(path):
            size = os.stat(path).st_size
            if size > 0:
                valid_files.append(path)
    return valid_files

# check if a file is existing and not empty
def file_is_valid(path):
    if os.path.isfile(path):
        size = os.stat(path).st_size
        if size > 0:
            return True
    return False

# create a tempfile class to use as object
class TempFile:
    def __init__(self, final_path: str, file_ext: str = None):
        self.final_path = final_path
        self.file_ext = file_ext
        os.makedirs(os.path.join(os.path.realpath(os.path.dirname(__file__)), "temp"), exist_ok=True)
        self.temp_file = tempfile.NamedTemporaryFile(dir=os.path.join(
            os.path.realpath(os.path.dirname(__file__)), "temp"),
            delete=False, suffix=file_ext)

    # return the actual path of the file
    def getname(self):
        if os.path.isfile(self.temp_file.name):
            return self.temp_file.name
        elif file_is_valid(self.final_path):
            return self.final_path
        else:
            return None
        
    # return the actual path of the file if not empty
    def getvalidname(self):
        if file_is_valid(self.temp_file.name):
            return self.temp_file.name
        elif file_is_valid(self.final_path):
            return self.final_path
        else:
            return None

    # save the temp file into final path
    def save(self, overwrite_if_valid: bool = True, update_path: str = None):
        # update final path case user specifies it
        if update_path is None:
            path = self.final_path
        else:
            path = update_path

        try:
            # if file not valid ou overwrite is enabled, move overwiting existing file
            if not file_is_valid(self.final_path) or overwrite_if_valid:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                shutil.move(self.temp_file.name, path)
                self.final_path = path
        except Exception as e:
            print(f"Error saving file: {e}")
            return

    # delete the file
    def destroy(self):
        try:
            # destroy temporary file if it exists
            if os.path.isfile(self.temp_file.name):
                os.remove(self.temp_file.name)
        except Exception:
            return
        
# copy an source file to destination if destination file is not equals to source
def copy_file_if_different(src_file, dst_file, silent: bool = False):
    if file_is_valid(dst_file):
    # Check if destination file exists and is different from source file
        if filecmp.cmp(src_file, dst_file) and not silent:
            print(f"{dst_file} already exists and is the same. No need to copy.")
            return

    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    shutil.copyfile(src_file, dst_file)
    if not silent:
        print(f"copied to {dst_file}")

# function to delete dir and all its content using shutil
def delete_folder(path):
    shutil.rmtree(path)
