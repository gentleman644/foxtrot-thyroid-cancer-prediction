import tkinter as tk
from tkinter import ttk
from ui.Thyroid_Cancer_Model import thyroid_Cancer_Model

class mainWindow():
    #Purpose: creates and controls the main window the user will see in the screen
    #Author: Tim Liu
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('demo')
        self.root.config(bg = 'gray', width=800)
        self.root.geometry('1000x800+50+50') #you may want to change this to actual numbers
        self.frames = []
        self.previous_frame = None
        self.dropmenu = []
        self.model_frames  = []
        self.model = thyroid_Cancer_Model()
        self._create_frames()
        self._create_dropmenu()

    def _create_frames(self):
        #Purpose: creates all the frames for the main window
        #Author: Tim Liu
        first_frame = tk.Frame(self.root,
                               relief='solid',
                               highlightbackground="black",
                               highlightthickness= 2)
        first_frame.pack(side = 'top', padx=10, pady=10, expand=True)
        label = tk.Label(first_frame, text='MACHINE MODEL',font=("Arial", 16, "bold"), width=50, height=3)
        label.pack()
        self.frames.append(first_frame)

        second_frame = self.model.make_ui(self.root)
        self.frames.append(second_frame)
        self.previous_frame = self.model.show_ui(self.frames[0])

    def _create_dropmenu(self):
        #Purpose: creates the option frame with all the buttons to different models
        #Author: Tim Liu
        menu_list = tk.Frame(self.root, bg='white')
        button_1 = tk.Button(menu_list, command = lambda: self._change_frame(self.model), text= 'Option 1', relief = 'flat')
        button_1.pack(fill = 'x')
        self.dropmenu.append(menu_list)
        self.dropmenu.append(button_1)

        def toggle_dropdown():
            #Purpose: shows the all options in the option list or not show them
            #Author Tim Liu
            if self.dropmenu[0].winfo_ismapped():
                self.dropmenu[0].pack_forget()
            else:
                self.dropmenu[0].pack(anchor='nw', side='top', padx=10, pady=10)

        menu_button = tk.Button(self.root, text='☰', relief = 'flat', command = toggle_dropdown)
        menu_button.pack(anchor='nw', side='top', padx=10, pady=10)

    def _change_frame(self, model_class):
        #Purpose: changes the current model frame into a different model frame
        #Parameter: model class
        #Author: Tim Liu
        self.previous_frame.pack_forget()
        self.previous_frame = model_class.show_ui(self.frames[0])
        self.previous_frame.pack()

    def run(self):
        self.root.mainloop()