import json
import os



class JSONHandler:



    def __init__(self, file_path, wanna_print = False):
        """
        Initialize with the path to the JSON file.
        """
        self.file_path = file_path
        self.wanna_print = wanna_print
        if not os.path.exists(self.file_path):
            self.reset()



    def subscribe_all(self, new_complete_list: list):
        '''
        Delete the whole file, and substitute it with the passed argument
        '''
        # Ensure the passed argument is of type list
        assert isinstance(new_complete_list, list), "Argument 'new_complete_list' must be of type list"
        # Reset the file, but re-create it
        self.reset(re_create=True)
        # Add all the elements the the new file
        for element in new_complete_list:
            self.add_dictionary(element)



    def reset(self, re_create = False):
        """
        Delete the file if it exists, and create a new empty JSON file with an empty list.
        """
        # Delete the file if it already exists
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            if self.wanna_print: print(f"File '{self.file_path}' has been deleted.")
        
        if re_create:
            # Create a new empty JSON file with an empty list
            with open(self.file_path, 'w') as file:
                json.dump([], file, indent=4)
            if self.wanna_print: print(f"New empty JSON file created at '{self.file_path}'.")



    def add_dictionary(self, new_dict):
        """
        Read the JSON file, add the new dictionary, and save it back.
        """
        # Read the current list of dictionaries from the file
        data = self.read_json_file()
        
        # Add the new dictionary to the list
        data.append(new_dict)

        # Write the updated list back to the file
        with open(self.file_path, 'w') as file:
            json.dump(data, file, indent=4)
        if self.wanna_print: print(f"Added new dictionary: {new_dict}")



    def update_dictionary(self, new_data):
        """
        Update the last dictionary in the JSON file with new keys and values.
        """
        # Read the current list of dictionaries from the file
        data = self.read_json_file()

        # Check if the file contains at least one dictionary
        if not data:
            if self.wanna_print:
                print("Could not find a dictionary...")
            return

        # Update the last dictionary with the new data
        data[-1].update(new_data)

        # Write the updated list back to the file
        with open(self.file_path, 'w') as file:
            json.dump(data, file, indent=4)
        if self.wanna_print: print(f"Updated the last dictionary with: {new_data}")



    def get_parameter(self, key):
        """
        Return a list of values for the specified key across all dictionaries in the JSON file.
        """
        data = self.read_json_file()
        # Collect values for the specified key across all dictionaries, if present
        return [d[key] for d in data if key in d]



    def read_json_file(self):
        """
        Helper function to read the JSON file and return the content.
        """
        try:
            with open(self.file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            # If the file doesn't exist, return an empty list
            return []
        except json.JSONDecodeError:
            # If the file is empty or corrupted, also return an empty list
            return []
