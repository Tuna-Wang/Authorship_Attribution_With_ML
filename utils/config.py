# -*- coding: utf-8 -*-

import os
from configparser import ConfigParser

from utils.const import CONFIG_FILE


class Config:
    CONFIG_PARSER = ConfigParser()
    CONFIG_PARSER.read(CONFIG_FILE)

    @staticmethod
    def get_source_data_path():
        try:
            data_path = Config.CONFIG_PARSER.get('dataset', 'dataset_folder')
        except Exception:
            return ''
        return data_path

    @staticmethod
    def get_workspace_folder():
        try:
            workspace_folder = Config.CONFIG_PARSER.get('common', 'workspace')
        except Exception:
            workspace_folder = 'workspace'
        os.makedirs(workspace_folder, exist_ok=True)
        return workspace_folder

    @staticmethod
    def get_log_folder():
        try:
            log_folder = os.path.join(Config.get_workspace_folder(), Config.CONFIG_PARSER.get('common', 'log_folder'))
        except Exception:
            log_folder = os.path.join(Config.get_workspace_folder(), 'logs')
        os.makedirs(log_folder, exist_ok=True)
        return log_folder
    
    @staticmethod
    def output_data_folder():
        try:
            processed_data= os.path.join(Config.get_workspace_folder(), Config.CONFIG_PARSER.get('common', 'processed_data'))
        except Exception:
            processed_data= os.path.join(Config.get_workspace_folder(), 'processed_data')
        os.makedirs(processed_data, exist_ok=True)
        return processed_data
    
    @staticmethod
    def model_save_path():
        try:
            model_save_path = os.path.join(Config.get_workspace_folder(), Config.CONFIG_PARSER.get('common', 'models'))
        except Exception:
            model_save_path = os.path.join(Config.get_workspace_folder(), 'models')
        os.makedirs(model_save_path, exist_ok=True)
        return model_save_path
    
    @staticmethod
    def result_save_path():
        try:
            result_save_path = os.path.join(Config.get_workspace_folder(), Config.CONFIG_PARSER.get('common', 'results'))
        except Exception:
            result_save_path = os.path.join(Config.get_workspace_folder(), 'results')
        os.makedirs(result_save_path, exist_ok=True)
        return result_save_path
