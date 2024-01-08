"""

    Given a class ID, returns the associated class name.

    :param class_id: An integer representing the class ID.
    :return: A string representing the class name.

"""
def get_class_name(class_id):

    class_labels = {
        0: 'Abstract_Expressionism',
        1: 'Action_painting',
        2: 'Analytical_Cubism',
        3: 'Art_Nouveau',
        4: 'Baroque',
        5: 'Color_Field_Painting',
        6: 'Contemporary_Realism',
        7: 'Cubism',
        8: 'Early_Renaissance',
        9: 'Expressionism',
        10: 'Fauvism',
        11: 'High_Renaissance',
        12: 'Impressionism',
        13: 'Mannerism_Late_Renaissance',
        14: 'Minimalism',
        15: 'Naive_Art_Primitivism',
        16: 'New_Realism',
        17: 'Northern_Renaissance',
        18: 'Pointillism',
        19: 'Pop_Art',
        20: 'Post_Impressionism',
        21: 'Realism',
        22: 'Rococo',
        23: 'Romanticism',
        24: 'Symbolism',
        25: 'Synthetic_Cubism',
        26: 'Ukiyo_e'
    }

    return class_labels.get(class_id, "Unknown Class")