import pandas as pd
import os

class DownloadData:
    def __init__(self, part="parta", save_folder="./data"):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self.part = part
        self.save_folder = save_folder
        if part not in ["parta", "partb", "partc", "partd", "parte", "partf", "partg", "parth", "parti", "partj", "all"]:
            raise Exception("Invalid part")
        
        self.commands = []
        
    def _download_data(self):
        if self.part == "all":
            path = os.path.join(self.save_folder, "youcook2_raw_videos.tar.gz")
            download_command = """wget -c "https://prism.eecs.umich.edu/natlouis/youcook2/raw_videos.tar.gz" \
           -O {path} --no-check-certificate""".format(path=path)
            extract_command = "tar -xf {path} -C {save_folder}".format(path=path, save_folder=self.save_folder)

        else:
            suffix = self.part.split("part")[-1]
            path = os.path.join(self.save_folder, "raw_videos.parta{suffix}".format(suffix=suffix))
            download_command = """wget -c "https://prism.eecs.umich.edu/natlouis/youcook2/raw_videos.parta{suffix}" \
            -O {path} --no-check-certificate""".format(suffix=suffix, path=path)
            extract_command = "tar -xf {path} -C {save_folder}".format(suffix=suffix, path=path, save_folder=self.save_folder)

        self.commands.extend([download_command, extract_command])


    def _download_label_type(self):
        path = os.path.join(self.save_folder, "label_foodtype.csv")
        download_command = """wget -c "http://youcook2.eecs.umich.edu/static/YouCookII/label_foodtype.csv" \
        -O {path} --no-check-certificate""".format(path=path)
        self.commands.append(download_command)

    def _download_annotations(self):
        train_path = os.path.join(self.save_folder, "youcook2_annotations_trainval.tar")
        test_path = os.path.join(self.save_folder, "youcook2_annotations_test.tar")

        download_command_train = """wget -c "http://youcook2.eecs.umich.edu/static/YouCookII/youcookii_annotations_trainval.tar.gz" \
        -O {train_path} --no-check-certificate""".format(train_path=train_path)

        download_command_test = """wget -c "http://youcook2.eecs.umich.edu/static/YouCookII/youcookii_annotations_test_segments_only.tar.gz" \
            -O {test_path} --no-check-certificate""".format(test_path = test_path)

        extract_command_train = "tar -xf {train_path} -C {save_folder}".format(train_path=train_path, save_folder=self.save_folder)
        extract_command_test = "tar -xf {test_path} -C {save_folder}".format(test_path = test_path, save_folder = self.save_folder)

        self.commands.extend([download_command_train,
                               download_command_test, 
                               extract_command_train, 
                               extract_command_test])
        
    def run(self):
        self._download_data()
        self._download_label_type()
        self._download_annotations()

        # Execute commands
        for cmd in self.commands:
            print(f"Running: {cmd}")
            os.system(cmd)
        


