## This a data processing worker that reads and filter data from the dataset.
# -*- coding: utf-8 -*-

from utils.config import Config
from utils.const import InputDataTitles
import pandas as pd
import os
from collections import defaultdict


class Worker:
    def __init__(self, logger):
        self.logger = logger
        self.data_path = os.path.join(Config.get_source_data_path(), 'script')
        
        
    def lines_extraction(self):
        '''
        This function will attribute the lines to its character.
        The input is the script file, output is a df with character as one column and lines as another column.
        '''

        labeled_data = defaultdict(list)

        # Read the script file
        with open(self.data_path,'r', encoding = 'utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                line = line.split(' ')
                if len(line) < 2:
                    continue
                speaker, text = line[0], line[1:]
                if speaker is None or text is None or text[0] is None:
                    continue
                try:
                    if speaker in InputDataTitles.CHARACTER_SET and text[0][0].isupper():
                        text = self._remove_brakcets(text)
                        labeled_data[speaker].append(' '.join(text))
                    # brackets are movement and emotion descriptions, we don't need it for this task
                    elif speaker in InputDataTitles.CHARACTER_SET and text[0][0] == '(':
                        i = 0
                        while i < len(text) and text[i][-1] != ')':
                            i += 1
                        if i < len(text):
                            text = text[i+1:]
                            if text[0][0].isupper():
                                text = self._remove_brakcets(text)
                                labeled_data[speaker].append(' '.join(text))
                            else:
                                continue
                    # quotation marks in the script usually are still valid lines of a spaker, we keep this
                    elif speaker in InputDataTitles.CHARACTER_SET and (text[0][0] == '"' or text[0][0] == "'"):
                         text = self._remove_brakcets(text)
                         labeled_data[speaker].append(' '.join(text))
                except Exception as e:
                    self.logger.error(f"Error processing line: {line}. Error: {str(e)}")

        # log info to document data size for each speaker
        self.logger.info('Total %d lines extracted for character %s' % (len(labeled_data[InputDataTitles.CHARACTER_H]), InputDataTitles.CHARACTER_H))  
        self.logger.info('Total %d lines extracted for character %s' % (len(labeled_data[InputDataTitles.CHARACTER_B]), InputDataTitles.CHARACTER_B))
        self.logger.info('Total %d lines extracted for character %s' % (len(labeled_data[InputDataTitles.CHARACTER_J]), InputDataTitles.CHARACTER_J)) 

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame.from_dict(labeled_data, orient='index').transpose()
        df = df.melt(var_name = 'character', value_name= 'lines')
        df = df.dropna()
        df = df.reset_index(drop=True)
        df.columns = ['label', 'text']

        # Store the DataFrame in a CSV file
        output_path = os.path.join(Config.output_data_folder(), 'lines_extraction.csv')
        df.to_csv(output_path, index=False, encoding='utf-8')

        return df
                
    def _remove_brakcets(self,splitted_line):
        '''
        This helper function will remove the content in the brackets.
        args:
            splitted_line: the line that is splitted by space.
        return:
            the line without the content in the brackets, still splitted by space.
        '''
        word_stack = []
        for word in splitted_line:
            if '(' in word and ')' in word:
                # Skip words that contain both '(' and ')'
                continue
            elif word[-1] == ')' and word_stack:
                # Remove words until the matching '(' is found
                while word_stack and '(' not in word_stack[-1]:
                    word_stack.pop()
                if word_stack and '(' in word_stack[-1]:
                    word_stack.pop()
            else:
                word_stack.append(word)
        return word_stack

   