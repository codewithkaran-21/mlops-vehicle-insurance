import sys 
import os 
import logging

def error_messsage_detail(error : Exception , error_detail : sys) -> str:
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    error_message = f"Error occured in Python Script : [{file_name}] at line number : [{line_no}] error message : {str(error)}"
    logging.error(error_message)

    return error_message

class MyEXception(Exception):
    def __init__(self , error_message : str , error_detail : sys):
        super().__init__(error_message)

        self.error_message  = error_messsage_detail(error_message , error_detail)

        def __str__(self):
            return self.error_message