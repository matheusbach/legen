import filecmp
import os
import shutil
import tempfile
from inspect import currentframe, getframeinfo
from pathlib import Path

# return the itens in array that is not inexisting or empty
def validate_files(paths):
    valid_files = [path for path in paths if file_is_valid(path)]
    return valid_files

# check if a file is existing and not empty
def file_is_valid(path):
    if path is not None and path.is_file():
        size = path.stat().st_size
        if size > 0:
            return True
    return False

# validate if an string is a valid dir or path with valid content
def check_valid_path(path_str):
    # create a Path object from the input string
    path = Path(path_str)

    # check if the path exists
    if not path.exists():
        raise FileNotFoundError(f"The path '{path_str}' does not exist.")

    # check if it's a directory with at least one file or a valid file with content
    if path.is_dir():
        # check if the directory has at least one file with content
        files_with_content = [file for file in path.iterdir() if file.is_file() and file.stat().st_size > 0]
        if not files_with_content:
            raise ValueError(f"The directory '{path_str}' does not contain any files with content.")
    elif path.is_file():
        # check if the file has content
        if path.stat().st_size == 0:
            raise ValueError(f"The file '{path_str}' does not contain any content.")
    else:
        raise ValueError(f"The path '{path_str}' is neither a valid directory nor a valid file.")

    return path

# validate if an string is a valid dir or path with valid content
def check_existing_path(path_str):
    # create a Path object from the input string
    path = Path(path_str)

    # check if the path exists
    if not path.exists():
        raise FileNotFoundError(f"The path '{path_str}' does not exist.")

    # check if it's a directory with at least one file or a valid file with content
    if not path.is_dir() and not path.is_file():
        raise ValueError(f"The path '{path_str}' is neither a valid directory nor a valid file.")

    return path_str

# create a tempfile class to use as object
class TempFile:

    def __init__(self, final_path: Path, file_ext: str = None):
        self.final_path: Path = None if final_path is None else Path(
            final_path)
        self.file_ext = file_ext
        os.makedirs(Path(Path(getframeinfo(
            currentframe()).filename).resolve().parent, "temp"), exist_ok=True)
        self.temp_file: tempfile.NamedTemporaryFile = tempfile.NamedTemporaryFile(dir=Path(Path(getframeinfo(currentframe()).filename).resolve().parent, "temp"),
                                                                                  delete=False, suffix=file_ext)

        self.temp_file_name = self.temp_file.name
        self.temp_file_path: Path = Path(self.temp_file.name)

        self.temp_file.close()

    # return the actual path of the file
    def getpath(self):
        if self.temp_file_path.is_file():
            return self.temp_file_path
        elif file_is_valid(self.final_path):
            return self.final_path
        else:
            return None

    # return the actual path of the file if not empty
    def getvalidpath(self):
        if file_is_valid(self.temp_file_path):
            return self.temp_file_path
        elif file_is_valid(self.final_path):
            return self.final_path
        else:
            return None

    # save the temp file into final path
    def save(self, overwrite_if_valid: bool = True, update_path: Path = None):
        # update final path case user specifies it
        if update_path is None:
            path: Path = self.final_path
        else:
            path: Path = update_path

        try:
            # if file not valid ou overwrite is enabled, move overwiting existing file
            if not file_is_valid(self.final_path) or overwrite_if_valid:
                os.makedirs(path.parent, exist_ok=True)
                shutil.move(self.temp_file_path, path)
                self.final_path = path
        except Exception as e:
            print(f"Error saving file: {e}")
            return

    # delete the file
    def destroy(self):
        try:
            # destroy temporary file if it exists
            if file_is_valid(self.temp_file_path):
                os.remove(self.temp_file_path)
        except Exception:
            return

# copy an source file to destination if destination file is not equals to source
def copy_file_if_different(src_file: Path, dst_file: Path, silent: bool = False):
    if file_is_valid(dst_file):
        # Check if destination file exists and is different from source file
        if filecmp.cmp(src_file, dst_file) and not silent:
            print(f"{dst_file} already exists and is the same. No need to copy.")
            return

    os.makedirs(dst_file.parent, exist_ok=True)
    shutil.copyfile(src_file, dst_file)
    if not silent:
        print(f"copied to {dst_file}")

# function to delete dir and all its content using shutil
def delete_folder(path: Path):
    if path.is_dir():
        shutil.rmtree(path)


def update_folder_times(folder_path):
    folder_path = Path(folder_path)

    # Keep track of the newest file's modification time for this folder
    newest_file_time = None

    for item in folder_path.iterdir():
        # Check if the item is a file
        if item.is_file():
            file_time = item.stat().st_mtime  # Get modification time of the file

            # Update the newest_file_time if it's the first file or if the current file is newer
            if newest_file_time is None or file_time > newest_file_time:
                newest_file_time = file_time

        # If the item is a subfolder, recursively update its times and find its newest file time
        elif item.is_dir():
            subfolder_newest_time = update_folder_times(item)

            # Update the newest_file_time if a subfolder's newest file is newer
            if newest_file_time is None or (subfolder_newest_time is not None and subfolder_newest_time > newest_file_time):
                newest_file_time = subfolder_newest_time

    # Update the folder's modification and creation times with the newest_file_time
    if newest_file_time is not None:
        # Convert to an integer (necessary on some systems)
        newest_file_time = int(newest_file_time)
        os.utime(path=folder_path, times=(newest_file_time, newest_file_time))

    return newest_file_time  # Return the newest_file_time to update the parent folder's time
