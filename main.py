#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import pandas as pd
import joblib
# import numpy as np
from logging.config import fileConfig
from data_processor.worker import Worker as data_processor
from feature_extractor.features import FeatureGetter
from classifiers.ngram import NgramModel
# from classifiers.cnn import CNNClassifier
from classifiers.unified_parser import TextDataset
from classifiers.cnn_new import CNNModel as NewCNNClassifier

from classifiers.logistic import LogisticClassifier
from evaluation.evaluator import Evaluator
import csv
import re


from utils.const import LOG_CONFIG_FILE
from utils.config import Config
from utils.const import InputDataTitles



if __name__ == '__main__':
    # Prepare logging
    config = {'debug_logfile': os.path.join(Config.get_log_folder(), 'main.log')}
    fileConfig(LOG_CONFIG_FILE, config)
    logger = logging.getLogger()

    # Load the raw data
    if not os.path.exists(Config.get_source_data_path()):
        logger.error('No data found in the source data path.')
        
    # if the processed data file is already created then skip the data processing
    if os.path.exists(os.path.join(Config.output_data_folder(), 'lines_extraction.csv')):
        logger.info('Processed data file already exists. Skipping data processing.')
        labeled_data_df = pd.read_csv(os.path.join(Config.output_data_folder(), 'lines_extraction.csv'))
    else:
        data_label = data_processor(logger)
        labeled_data_df = data_label.lines_extraction()


    # Create evaluator instance due to it is a shared function
    evaluate_model = Evaluator(logger)

    
    ##### LOGISTIC PART #####
    # We do not check if the model is already trained, we always retrain the model
    # Prepare feature vectors for the baseline logistic regression model
    if os.path.exists(os.path.join(Config.output_data_folder(), 'features.csv')):
        logger.info('Feature file already exists. Skipping feature extraction.')
        feature_df = pd.read_csv(os.path.join(Config.output_data_folder(), 'features.csv'))
    else:
        features_getter = FeatureGetter(logger, labeled_data_df)
        feature_df = features_getter.add_store_features()

    # Train and evaluate the logistic regression model
    logistic_model = LogisticClassifier(logger, feature_df, labeled_data_df)
    evaluate_model = Evaluator(logger)
    # Baseline model with manual feature selection
    baseline_model, baseline_test_results = logistic_model.train_model_feature()
    evaluator_results = evaluate_model.evaluate(baseline_model, baseline_test_results, None, None)
    # Optimized model using BERT embeddings
    bert_model, bert_test_results = logistic_model.train_model_bert()
    bert_evaluator_results = evaluate_model.evaluate(bert_model, bert_test_results, None, None)
   

    ##### NGRAM PART #####

    dataset = TextDataset(logger, labeled_data_df, 10)
    train_dataset, test_dataset = dataset.split_data(train_size=0.8)
    ngram_model = NgramModel(logger, labeled_data_df, ngram_range=(1, 6), max_features=500, train_dataset=train_dataset, test_dataset=test_dataset)
    logger.info('length of the ngram is 1 - 6')
    n_gram_trained_model, _ , _ = ngram_model.train_ngram_model(print_features=True)
    n_gram_predictions, n_gram_ground_truth = ngram_model.predict()
    ngram_model.store_result()
    # evaluate the ngram model
    evaluator_results = evaluate_model.evaluate(n_gram_trained_model, None, n_gram_predictions, n_gram_ground_truth)
    
    ##### CNN PART #####
    # Create consistent data splits for fair comparison
    main_dataset = TextDataset(logger, labeled_data_df, 10)
    main_train, main_test = main_dataset.split_data(train_size=0.8)
    
    # Split training data into train/val for CNN
    cnn_temp_dataset = TextDataset(logger, main_train, 10)
    cnn_train, cnn_val = cnn_temp_dataset.split_data(train_size=0.8)
    cnn_train = TextDataset(logger, cnn_train, 10)
    cnn_val = TextDataset(logger, cnn_val, 10)


    # Reduce model complexity to prevent overfitting
    cnn_model = NewCNNClassifier(
        vocab_size=len(cnn_train.vocab),
        embed_dim=50,      
        num_filters=50,   
        filter_sizes=[3,4,5],
        num_classes=len(cnn_train.label_encoder.classes_),
        dropout=0.7       
    )

    cnn_model.train_model(
        train_dataset=cnn_train,
        val_dataset=cnn_val,
        epochs=20,           
        batch_size=16,       
        learning_rate=0.01,  
        device='cpu',
        patience=5           
    )
    
    # Predict on test dataset and store results
    cnn_predictions, cnn_ground_truth, cnn_dict = cnn_model.predict_and_store_results(
        test_dataset=main_test, 
        vocab=cnn_train.vocab, 
        label_encoder=cnn_train.label_encoder,
        device='cpu'
    )
    
    # Evaluate the CNN model
    cnn_model.set_classes(cnn_train.label_encoder)
    cnn_evaluator_results = evaluate_model.evaluate(cnn_model, None, cnn_predictions, cnn_ground_truth)
        
