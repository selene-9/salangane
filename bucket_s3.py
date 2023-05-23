from typing import Dict

import aws_lib


def aws_upload(data: Dict):
    database = aws_lib.connect("AKIA4N27N62BP5RUYYIK", "8C6hauR6nNUj83VTuG65MSVqrSBPkK0VpZnSH/YR")
    database.push(data)
