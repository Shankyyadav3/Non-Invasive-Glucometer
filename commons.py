import numpy as np
from numpy import loadtxt, absolute
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pickle, csv, os, constants, math
import pandas as pd
import os, fnmatch
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from itertools import zip_longest


def save_labels_as_pdf(labels,name):
    # filename = filedialog.asksaveasfilename(defaultextension='.pdf')
    # if filename:
    filename = f"{name}_report.pdf"
    save_path = filedialog.asksaveasfilename(defaultextension=".pdf", initialfile=filename)
    if save_path:

        doc = SimpleDocTemplate(save_path, pagesize=letter)
        styles = getSampleStyleSheet()

        content = []
        for label in labels:
            text = label.cget("text")
            if text.find("Patient") or text.find("Final"):
                paragraph = Paragraph(text, styles['Normal'])
            else:
                paragraph = Paragraph(text, styles['Normal'])
            content.append(paragraph)
            content.append(Spacer(1, 12))  # Adjust spacing between paragraphs

        doc.build(content)


def project_decision_in_app(decision, name, body_vitals, bgl):
    # Create the main window
    window = tk.Tk()

    # Set the window title
    window.title("Non-Invasive Glucometer Prediction Result")

    # Create a label widget for the title
    title_label = tk.Label(window, text="Non-Invasive Glucometer Prediction Result", font=("Arial", 16, "bold"))
    title_label.pack(pady=10)

    gender = ''
    if body_vitals[1]=='0':
        gender = 'Male'
    elif body_vitals[1]=='1':
        gender = 'Female'

    # Create a label widget to display the message
    label1 = tk.Label(window, text=f"=> Patient Details :  ", font=("Arial", 16, 'underline'))
    label1.pack(padx=5, pady=5, anchor= 'w')
    label2 = tk.Label(window, text=f"    Name :- {name} ", font=("Arial", 16))
    label2.pack(padx=5, pady=5, anchor= 'w')
    label3 = tk.Label(window, text=f"    Age :- {body_vitals[0]} ", font=("Arial", 16))
    label3.pack(padx=5, pady=5, anchor= 'w')
    label4 = tk.Label(window, text=f"    Gender :- {gender} ", font=("Arial", 16))
    label4.pack(padx=5, pady=5, anchor= 'w')
    label5 = tk.Label(window, text=f"    Heart Rate :- {body_vitals[2]} ", font=("Arial", 16))
    label5.pack(padx=5, pady=5, anchor= 'w')
    label6 = tk.Label(window, text=f"    SPO2 :- {body_vitals[3]} ", font=("Arial", 16))
    label6.pack(padx=5, pady=5, anchor= 'w')
    label7 = tk.Label(window, text=f"    Blood Pressure :- {body_vitals[4]} / {body_vitals[5]} ", font=("Arial", 16))
    label7.pack(padx=5, pady=5, anchor= 'w')
    label8 = tk.Label(window, text=f"=> Final Results : ", font=("Arial", 16, 'underline'))
    label8.pack(padx=5, pady=5, anchor= 'w')
    label9 = tk.Label(window, text=f"    {name}, {decision} ", font=("Arial", 16))
    label9.pack(padx=5, pady=5, anchor= 'w')
    if bgl != 0:
        label10 = tk.Label(window, text=f"    Blood Glucose Level :- {bgl} ", font=("Arial", 16))
        label10.pack(padx=5, pady=5, anchor= 'w')
    label11 = tk.Label(window, text='' )
    label11.pack(padx=5, pady=5)


    if bgl!=0:
        labels = [title_label,label1,label2,label3,label4,label5,label6,label7,label8,label9,label10,label11]
    else:
        labels = [title_label,label1,label2,label3,label4,label5,label6,label7,label8,label9,label11]

    #Save button
    save_button = tk.Button(window, text="Save as PDF", command= lambda: save_labels_as_pdf(labels,name))
    save_button.pack()

    #Set the geometry
    app_width= 600
    app_height= 500

    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    x= (screen_width/2) - (app_width/2)
    y= (screen_height/2) - (app_height/2)
    window.geometry(f'{app_width}x{app_height}+{int(x)}+{int(y)}')
    # window.eval('tk::PlaceWindow . center')

    #Make the window jump above all
    # window.attributes('-topmost',True)

    # Start the main event loop
    window.mainloop()
    return

def fetch_test_data_sheet(test_data_dir):
    for root, dirs, files in os.walk(test_data_dir):
        for file in files:
            if fnmatch.fnmatch(file, '*.csv'):
                return os.path.join(root, file)


def get_body_vitals_info():
    # Get user input for age, gender, and heart rate
    age = int(input("Enter age: "))
    gender = input("Enter gender - Press 1 for Female and 0 for Male: ")
    heart_rate = int(input("Enter heart rate: "))
    max_bp = int(input("Enter Max BP: "))
    min_bp = int(input("Enter Min BP: "))
    spo2 = int(input("Enter SPO2 value: "))
    return [age, gender, heart_rate, spo2, max_bp, min_bp]

def bgl_test():
    ans = input("Does the patient want to have BGL(glucometer) test? Yes-(1) / No-(0) : ")
    if ans =='1':
        bgl=input('Enter BGL: ')
    else:
        bgl=0
    return bgl

def add_body_vitals_data_to_sensors_data(sensor_data_sheet, body_vitals, complete_data_sheet):
    with open(sensor_data_sheet, 'r') as file:
        reader = csv.reader(file)
        existing_data = list(reader)
    file.close()
    # Modify the data by adding new columns
    header = existing_data[0]
    data = existing_data[1:]

    # Add new column headers
    new_column_headers = ['Age', 'Gender', 'Heart_Beat', 'SPO2', 'max_BP', 'min_BP', 'Sample_No']
    header.extend(new_column_headers)
    body_vitals.append(1000)
    # Add new column data
    for row in data:
        # Example: Add values to new columns based on existing data
        row.extend(body_vitals)

    # Write the updated data to the new CSV file
    with open(complete_data_sheet, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([header] + data)
    file.close()
    return

def remove_irrelevant_data_test(test_data_sheet):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(test_data_sheet)

    # Remove the first 3 rows
    df = df.iloc[4:]

    # Remove the first 3 columns
    df = df.iloc[:, 3:]

    # New header names
    new_headers = ['Humidity', 'SSID', 'Temperature', 'Device', 'TGS2603', 'TGS2620', 'TGS2602', 'TGS2610', 'TGS2600', 'TGS822', 'TGS826',
               'MQ138', 'Timestamps']

    # Replace the headers
    df.columns = new_headers

    # Remove the SSID and Device
    df = df.drop('SSID', axis=1)
    df = df.drop('Device', axis=1)
    # Write the modified DataFrame back to a new CSV file
    new_csv_path = 'cleaned_test_data.csv'
    df.to_csv(new_csv_path, index=False)
    return new_csv_path

def get_dataset(data_sheet):
    # load data
    return loadtxt(data_sheet, delimiter=",", skiprows=1)  # skiprows used as row 0 contains headers


def get_data_as_X_Y(data_sheet):
    dataset = get_dataset(data_sheet)
    # split data into X and y
    X = dataset[:, 0:18]  # For PIMA's Dataset: dataset[:, 0:8]
    Y = dataset[:, 19]  ## For PIMA's Dataset: dataset[:, 8]
    return X, Y


def get_data_as_data_frame(data_sheet, skip_rows):
    return pd.read_csv(data_sheet, skiprows=skip_rows)


def get_csv_column_vals_using_names(data_sheet, column_names):
    df = get_data_as_data_frame(data_sheet, None)
    # get the values of multiple columns by column names
    return df[column_names].values


def compute_duration_from_timestamps(time_stamps):
    # convert timestamp column to datetime format
    timestamps = pd.to_datetime(time_stamps)
    # get the duration in seconds
    return (timestamps.max() - timestamps.min()).total_seconds() / 60


def get_aggregate_vals(patient_df, func):
    column_names = constants.AGGREGATE_DATA_COLUMN_NAMES
    aggregate_feature_vals = []
    for column_name in column_names:
        if func == "max":
            aggregate_feature_vals.append(np.max(patient_df[column_name]))
        elif func == "min":
            aggregate_feature_vals.append(np.min(patient_df[column_name]))
        elif func == "mean":
            aggregate_feature_vals.append(np.mean(patient_df[column_name]))
        elif func == "median":
            aggregate_feature_vals.append(np.median(patient_df[column_name]))
        else:
            print("incorrect function")
    return aggregate_feature_vals


def remove_nan_values(data):
    for i in range(0, len(data)):
        for j in range(0, len(data[i])):
            if math.isnan(data[i][j]):
                data[i][j] = 0.0
    return data


def save_model(model, file_name):
    print("Saving model: " + file_name)
    pickle.dump(model, open(file_name, 'wb'))
    return


def has_fractional_values(lst):
    for num in lst:
        if isinstance(num, float) and num % 1 != 0:
            return True
    return False


def get_train_test_split(X, Y):
    # split data into train and test sets
    test_size = 0.20
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    return X_train, X_test, y_train, y_test


def fetch_model_paths(model_dir):
    model_paths = []
    for file in os.listdir(model_dir):
        if file.endswith('.pkl'):
            file_path = os.path.join(model_dir, file)
            model_paths.append(file_path)
    return model_paths


def perform_padding(data):
    padded_list = []
    # Find the maximum length of sublists
    max_length = max(len(sublist) for sublist in data)
    for sub_list in data:
        extra = max_length - len(sub_list)
        if extra > 0:
            padding = [0.0] * extra
            sub_list = sub_list + padding
        padded_list.append(sub_list)
    return padded_list


def show_scores(y_true, y_pred):
    accuracy = f1_score_val = roc_area = 0
    try:
        accuracy = accuracy_score(y_true, y_pred) * 100.0
        f1_score_val = f1_score(y_true, y_pred) * 100.0
        roc_area = roc_auc_score(y_true, y_pred) * 100.0
        print("Accuracy: %.2f%%" % accuracy)  # 74.02%
        print("F1 score: %.2f%%" % f1_score_val)
        print("ROC score: %.2f%%" % roc_area)
    except:
        print("not possible to compute metrics")
    return [accuracy, f1_score_val, roc_area]


def get_mae_cross_val_score(model, X, Y, kfold):
    mae_scores = absolute(cross_val_score(model, X, Y, cv=kfold, scoring='neg_mean_absolute_error'))
    rmse_scores = absolute(
        cross_val_score(model, X, Y, cv=kfold, scoring='neg_root_mean_squared_error'))  # root_mean_Score_error
    mse_scores = absolute(
        cross_val_score(model, X, Y, cv=kfold, scoring='neg_mean_squared_error'))  # root_mean_Score_error
    msle_scores = absolute(
        cross_val_score(model, X, Y, cv=kfold, scoring='neg_mean_squared_log_error'))  # root_mean_Score_error
    mdae_scores = absolute(
        cross_val_score(model, X, Y, cv=kfold, scoring='neg_median_absolute_error'))  # root_mean_Score_error
    mpd_scores = absolute(
        cross_val_score(model, X, Y, cv=kfold, scoring='neg_mean_poisson_deviance'))  # root_mean_Score_error
    results = [mae_scores.mean(), mae_scores.std(), rmse_scores.mean(), rmse_scores.std(), mse_scores.mean(),
               mse_scores.std(), msle_scores.mean(), msle_scores.std(), mdae_scores.mean(), mdae_scores.std(),
               mpd_scores.mean(), mdae_scores.std()]
    return results


def get_norm_values(values_list):
    norm_values = []
    max_value = max(values_list)
    for elem in values_list:
        norm_values.append(elem / max_value)
    return norm_values


def get_norm_voltage_values(volt, temp, humid):
    norm_volt = []
    for i in range(0, len(volt)):
        norm_volt.append(volt[i] * temp[i] * humid[i])
    return norm_volt


def obtain_sample_specific_sensor_data(sensors_data, sample_id_list, sample_id):
    sample_specific_data = []
    for i in range(0, len(sample_id_list)):
        if sample_id_list[i] == sample_id:
            sample_specific_data.append(sensors_data[i])
    return sample_specific_data


def obtain_combined_sensors_volt_data(relevant_sensors_data_sheets):
    comb_rel_sensor_voltages = []
    for i in range(len(relevant_sensors_data_sheets[0])):  # loop for sensors
        sensor_volt = relevant_sensors_data_sheets[0][i]
        for j in range(1, len(relevant_sensors_data_sheets)):  # loop for acetone conc
            sensor_volt = sensor_volt + relevant_sensors_data_sheets[j][i]
        comb_rel_sensor_voltages.append(sensor_volt)
    return comb_rel_sensor_voltages


def obtain_combined_acetone_conc_data(relevant_sensors_data_sheets, acetone_conc_list):
    comb_rel_acetone_conc = []
    for i in range(len(relevant_sensors_data_sheets[0])):  # loop for sensors
        sensor_acetone_conc_list = []
        for j in range(0, len(relevant_sensors_data_sheets)):  # loop for acetone conc
            sensor_acetone_conc_list = sensor_acetone_conc_list + [acetone_conc_list[j]] * len(
                relevant_sensors_data_sheets[j][i])
        comb_rel_acetone_conc.append(sensor_acetone_conc_list)
    return comb_rel_acetone_conc


def compute_eval_metrics(y, y_fit, popt):
    residuals = y - y_fit
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    n = len(y)
    k = len(popt)
    aic = n * np.log(ss_res / n) + 2 * k
    bic = n * np.log(ss_res / n) + k * np.log(n)
    return [r2, aic, bic]


def get_sensor_data_per_conc(sensors_data, acetone_data, reqd_acetone_conc):
    sensor_data_per_conc = []
    acetone_conc_vals = constants.ACETONE_CONC
    for i in range(0, len(sensors_data)):
        data1 = sensors_data[:acetone_data.count(acetone_conc_vals[0])]
        endInd = len(data1) + acetone_data.count(acetone_conc_vals[1])
        data2 = sensors_data[len(data1):endInd]
        stInd = len(data1) + len(data2)
        endInd = stInd + acetone_data.count(acetone_conc_vals[2])
        data3 = sensors_data[stInd:endInd]
        stInd = len(data1) + len(data2) + len(data3)
        endInd = stInd + acetone_data.count(acetone_conc_vals[3])
        data4 = sensors_data[stInd:endInd]
        if reqd_acetone_conc == acetone_conc_vals[0]:
            reqd_data = data1
        elif reqd_acetone_conc == acetone_conc_vals[1]:
            reqd_data = data2
        elif reqd_acetone_conc == acetone_conc_vals[2]:
            reqd_data = data3
        else:
            reqd_data = data4
        sensor_data_per_conc.append(reqd_data)
    return sensor_data_per_conc


def fetch_model_name(model_path):
    model_name = model_path[model_path.rfind('/') + 1:model_path.rfind('.pkl')]
    return model_name


def array_to_csv(results, results_file_path):
    pd.DataFrame(np.array(results)).to_csv(results_file_path)
    return


def get_balanced_dataset(X, Y):
    new_X = []
    new_Y = []
    zero_count = Y.count(0)
    one_count = Y.count(1)
    flag = 0
    if one_count > zero_count:
        flag = 1
    # if flag = 1 means we need to take zero_count number of features for both X and Y
    count_0 = 0
    count_1 = 0
    index = 0
    if flag == 1:
        counter = zero_count
    else:
        counter = one_count
    while count_0 < counter or count_1 < counter:
        if Y[index] == 0:
            count_0 = count_0 + 1
        else:
            count_1 = count_1 + 1
        new_X.append(X[index])
        new_Y.append(Y[index])
        index = index + 1
    return new_X, new_Y


def get_file_paths(dir):
    return [os.path.join(root, name)
            for root, dirs, files in os.walk(dir)
            for name in files]


def get_three_class_labels(labels):
    bgl_labels = []
    for label in labels:
        if label < 70:
            bgl_label = 1
        elif 70 <= label <= 140:
            bgl_label = 0
        else:
            bgl_label = 2
        bgl_labels.append(bgl_label)
    return bgl_labels
