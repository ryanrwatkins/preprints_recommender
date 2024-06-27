""" biography = "I am a researcher in education and AI and other stuff"
adjacent_value = 2  # between 1 and 4  -- get from online version
discipline = "" """

import os

""" user_profile_path = os.path.join(os.getcwd(), 'src/user_profile.txt')  """  # for testing
""" user_profile_path = os.path.join(os.getcwd(), "user_profile.txt") """
directory_path = os.path.dirname(os.path.abspath(__file__))
user_profile_path = os.path.join(directory_path, "user_profile.txt")


class UserProfile:
    def __init__(self, biography, adjacent_value=None, discipline=None, keywords=None):
        self.biography = biography
        self.adjacent_value = adjacent_value
        self.discipline = discipline
        self.keywords = keywords

    def save(self):
        with open(user_profile_path, "w") as file:
            file.write(f"Biography: {self.biography}\n")
            file.write(f"Adjacent value: {self.adjacent_value}\n")
            file.write(f"Discipline: {self.discipline}\n")
            file.write(f"Keywords: {self.keywords}\n")
        print(
            f"User profile saved: {self.biography}, {self.adjacent_value}, {self.discipline},  {self.keywords}"
        )

    @classmethod
    def load(cls):
        if os.path.exists(user_profile_path):

            with open(user_profile_path, "r") as file:
                data = file.readlines()
                biography = data[0].split(": ")[1].strip()
                adjacent_value = data[1].split(": ")[1].strip()
                discipline = data[2].split(": ")[1].strip()
                keywords = data[3].split(": ")[1].strip()
                print(
                    f"User profile loaded: {biography},{adjacent_value}, {discipline},  {keywords}"
                )
                return cls(biography, adjacent_value, discipline, keywords)
        else:
            print("User profile file not found, loading default profile.")


# Global instance of UserProfile
user_profile = UserProfile.load()
