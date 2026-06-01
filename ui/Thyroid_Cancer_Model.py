import tkinter as tk
from tkinter import ttk
from ui.Base import base_model
from services.Model import model
from services.Age_Verification import int_verification
from services.T_Level_Verification import float_verification

class thyroid_Cancer_Model(base_model):
    # purpose: it holds the first iteration of the
    #          thyroid cancer prediction machine learning model
    # Author: Tim Liu
    def __init__(self):
        self.model = model()
        self.parts = []
        self.frame = None

    def make_ui(self, root):
        #purpose: creates a frame, which holds inputs towards our prediction model.
        #Return: frame
        #Author(s): Tim Liu
        frame = tk.Frame(root, width=850, height=800)
        frame.grid_propagate(False)
        self.frame = frame

        age_Text = tk.Label(frame, text='age:')
        age_Text.grid(row=0,column=0)
        int_proof = root.register(int_verification)
        age = tk.Entry(frame,
                       validate='key',
                       validatecommand=(int_proof, '%S'))
        age.grid(row=0, column=1)
        self.parts.append(age)

        smoking_Text = tk.Label(frame, text='smoking:')
        smoking_Text.grid(row=1,column=0)
        smoking = ttk.Combobox(frame, values=['True', 'False'], state='readonly')
        smoking.grid(row=1, column=1)
        self.parts.append(smoking)

        obesity_Text = tk.Label(frame, text='obesity:')
        obesity_Text.grid(row=2,column=0)
        obesity = ttk.Combobox(frame, values=['True', 'False'], state='readonly')
        obesity.grid(row=2,column=1)
        self.parts.append(obesity)

        family_History_Text = tk.Label(frame, text='family History:')
        family_History_Text.grid(row=3,column=0)
        family_History = ttk.Combobox(frame, values=['True', 'False'], state='readonly')
        family_History.grid(row=3,column=1)
        self.parts.append(family_History)

        gender_Text = tk.Label(frame,text='gender:')
        gender_Text.grid(row=4,column=0)
        gender = ttk.Combobox(frame, values=['Male', 'Female'], state='readonly')
        gender.grid(row=4,column=1)
        self.parts.append(gender)

        thyroid_Cancer_Risk_Text = tk.Label(frame,text='thyroid cancer risk:')
        thyroid_Cancer_Risk_Text.grid(row=5,column=0)
        thyroid_Cancer_Risk = gender = ttk.Combobox(frame, values=['High', 'Medium', 'Low'], state='readonly')
        thyroid_Cancer_Risk.grid(row=5,column=1)
        self.parts.append(thyroid_Cancer_Risk)

        T4_Level_Text = tk.Label(frame, text='T4 Level:')
        T4_Level_Text.grid(row=6,column=0)
        float_proof = root.register(float_verification)
        T4_Level = tk.Entry(frame,
                       validate='key',
                       validatecommand=(float_proof, '%P'))
        T4_Level.grid(row=6,column=1)
        self.parts.append(T4_Level)

        T3_Level_Text = tk.Label(frame, text='T3 Level:')
        T3_Level_Text.grid(row=7,column=0)
        T3_Level = tk.Entry(frame,
                       validate='key',
                       validatecommand=(float_proof, '%P'))
        T3_Level.grid(row=7,column=1)
        self.parts.append(T3_Level)

        TSH_Level_Text = tk.Label(frame, text='TSH Level:')
        TSH_Level_Text.grid(row=8,column=0)
        TSH_Level = tk.Entry(frame,
                       validate='key',
                       validatecommand=(float_proof, '%P'))
        TSH_Level.grid(row=8, column=1)
        self.parts.append(TSH_Level)

        nodule_Size_Text = tk.Label(frame, text='Nodule Size:')
        nodule_Size_Text.grid(row=9,column=0)
        nodule_Size = tk.Entry(frame,
                       validate='key',
                       validatecommand=(float_proof, '%P'))
        nodule_Size.grid(row=9,column=1)
        self.parts.append(nodule_Size)

        country_text = tk.Label(frame, text='Country:')
        country_text.grid(row=10,column=0)
        country_list = ['Germany', 'Nigeria', 'India', 'UK', 'South Korea', 'Brazil', 'China', 'US', 'Japan']
        country = ttk.Combobox(frame, values= country_list, state='readonly')
        country.grid(row=10,column=1)
        self.parts.append(country)

        ethnicity_text = tk.Label(frame, text='Ethnicity:')
        ethnicity_text.grid(row=11,column=0)
        ethnicity_list = ['Hispanic', 'Asian', 'African', 'Middle Eastern']
        ethnicity = ttk.Combobox(frame, values= ethnicity_list, state='readonly')
        ethnicity.grid(row=11,column=1)
        self.parts.append(ethnicity)

        radiation_Exposure_Text = tk.Label(frame, text='Radation Exposure')
        radiation_Exposure_Text.grid(row=12,column=0)
        radiation_Exposure = ttk.Combobox(frame, values=['True', 'False'], state='readonly')
        radiation_Exposure.grid(row=12,column=1)
        self.parts.append(radiation_Exposure)

        Iodine_Deficiency_Text = tk.Label(frame, text='Iodine Deficiency')
        Iodine_Deficiency_Text.grid(row=13,column=0)
        Iodine_Deficiency = ttk.Combobox(frame, values=['True', 'False'], state='readonly')
        Iodine_Deficiency.grid(row=13,column=1)
        self.parts.append(Iodine_Deficiency)

        Diabetes_Text = tk.Label(frame, text='Diabetes:')
        Diabetes_Text.grid(row=14,column=0)
        Diabetes = ttk.Combobox(frame, values=['True', 'False'], state='readonly')
        Diabetes.grid(row=14,column=1)
        self.parts.append(Diabetes)

        Submit_Button = tk.Button(frame, text='submit', command = lambda:self._update_model())
        Submit_Button.grid(row=15,column=0)

        final_text = tk.Label(frame, text='')
        final_text.grid(row=16,column=0)
        self.parts.append(final_text)
        return self.frame
        
    def show_ui(self, frame):
        #Purpose: will show the frame inside of the selected parent frame
        #Return: frame
        #Author: Tim Liu
        self.frame.pack(after=frame, side = 'right', padx=10, pady=10, expand=True)
        return self.frame
    
    def _update_model(self):
        #Purpose: It will update the parameters of our model and make the prediction
        #         based the inputs and show it into the output text
        #Return: None
        #Author: Tim Liu
        self.model.Age = self._update_int(0)
        self.model.Smoking = self._update_True_Or_False(1)
        self.model.Obesity = self._update_True_Or_False(2)
        self.model.Family_History = self._update_True_Or_False(3)
        self.model.Gender = self.parts[4].get()
        self.model.Thyroid_Cancer_Risk = self.parts[5].get()
        self.model.T4_Level = self._update_float(6)
        self.model.T3_Level = self._update_float(7)
        self.model.TSH_Level = self._update_float(8)
        self.model.Nodule_Size = self._update_float(9)
        self.model.Country = self.parts[10].get()
        self.model.Ethnicity = self.parts[11].get()
        self.model.Radiation_Exposure = self._update_True_Or_False(12)
        self.model.Iodine_Deficiency = self._update_True_Or_False(13)
        self.model.Diabetes = self._update_True_Or_False(14)

        self.parts[15].config(text = self.model.predict())

    def _update_True_Or_False(self, position):
        #Purpose: find out if the text from the inputs are True or False
        #Return: True or False
        #Author: Tim Liu
        try:
            if self.parts[position].get() == 'True':
                return True
            elif self.parts[position].get() == 'False':
                return False
        except:
            return False

    def _update_int(self, position):
        #Purpose: convert text into integer format
        #Return: integer
        #Author: Tim Liu
        try:
            return int(self.parts[position].get())
        except ValueError:
            return 0 #TODO: change this later

    def _update_float(self, position):
        #Purpose: convert text into float format
        #Return: float
        #Author: Tim Liu
        try:
            return float(self.parts[position].get())
        except ValueError:
            return 0.0 #TODO: change this later
    