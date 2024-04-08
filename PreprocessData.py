"""
This is the script to preprocess the bytecode dataset
"""

from xml.dom import minidom
import xml.etree.cElementTree as ET
import sys

input_filename = sys.argv[1]
output_filename = sys.argv[2]

def remove_comments(method):

    new_method = ""

    for line in method.splitlines(): 
        new_method += line.split("//")[0] + "\n"

    #Remove trailing whitespaces
    new_method = new_method.rstrip(" ")
    return new_method

def normalize_switch(method):
    new_method = ""
    inside_switch = False

    for line in method.splitlines(): 
        new_line = ""
        if inside_switch:
            if "}" in line:
                inside_switch = False
            else:
                new_line += "{switch_key}"
                new_line += ": {switch_block}"
                new_method += new_line + "\n"
        elif "tableswitch" in line:
            new_method += "tableswitch" + "\n"
        elif "lookupswitch" in line:
            new_method += "lookupswitch" + "\n"
        else:
            new_method += line + "\n"
        if "tableswitch" in line or "lookupswitch" in line:
            inside_switch = True
           
    return new_method

def normalize_exception(method):
    new_method = ""
    inside_exception = False
    is_labels = False

    for line in method.splitlines(): 
        new_line = ""
        if inside_exception:
            if not is_labels:
                if (line != "" and line != "}"):
                    new_line += "{from} {to} {target} "+line.split()[-1]
                else:
                    inside_exception = False
                    new_line = line
                new_method += new_line + "\n"
            else:
                is_labels = False
                new_method += line + "\n"
        else:
            new_method += line + "\n"
        if "Exception table:" in line:
            inside_exception = True
            is_labels = True
           
    return new_method

def normalize_arg(arg):
    if arg.startswith("#"):
        return "{reg}"
    elif arg.strip().strip('-').isnumeric():
        return "{const}"
    else: 
        return arg

def normalize_args(method):

    new_method = ""

    for line in method.splitlines()[2:]:

        line_arr = line.split()

        if (len(line_arr) > 2):
            
            if line_arr[2].endswith(","):
                line_arr[3] = normalize_arg(line_arr[3])

            line_arr[2] = normalize_arg(line_arr[2])
        
        new_method += ' '.join(line_arr) +"\n"

    return new_method

def remove_method_defs(file_str):
    new_method = ""
    for line in file_str.splitlines():
        if (len(line) > 0 and line[0].isdigit()):
            new_method += " ".join(line.split()[1:]) + "\n"
        elif line.startswith("{switch_key}") or line.startswith("tableswitch") or line.startswith("lookupswitch") or line.startswith("Exception table") or line.startswith("{from}"):
            new_method += line + "\n"
    return new_method

def preprocess_data(method):
    # return normalize_instructions(remove_method_defs(normalize_args(normalize_exception(normalize_switch(remove_comments(method))))))
    return remove_method_defs(normalize_args(normalize_exception(normalize_switch(remove_comments(method)))))

def parse_xml(file_name):
    
    parsed_file = None

    try:
        clones_file = open(file_name)
        parsed_file = minidom.parse(clones_file)

    finally:
        clones_file.close()

    return parsed_file

def preprocess_functions(clones_node):

    clones = ET.Element("clones")

    i = 0
    for clone_node in clones_node.childNodes:

        if clone_node.nodeType ==  1:
            clone = ET.SubElement(clones, "clone")
            similarity = clone_node.getAttribute("similarity")
            # if clone_node.getElementsByTagName("source1")[0].firstChild:
            #     print(clone_node.getElementsByTagName("source1")[0].firstChild.data)
            if not clone_node.getElementsByTagName("source1")[0].firstChild:
                print("clone_node.getElementsByTagName(source1)", clone_node.getElementsByTagName("source1")[0])
                print("similarity",similarity)
                print("i", i)
                break
            source1 = clone_node.getElementsByTagName("source1")[0].firstChild.data
            source2 = clone_node.getElementsByTagName("source2")[0].firstChild.data
            source1_p = preprocess_data(source1)
            source2_p = preprocess_data(source2)
            clone.attrib["similarity"] = similarity
            source1_node = ET.SubElement(clone, "source1")
            source2_node = ET.SubElement(clone, "source2")
            source1_node.text = source1_p
            source2_node.text = source2_p
            source1_node.attrib["file1"] = clone_node.getElementsByTagName("source1")[0].getAttribute("file")
            source2_node.attrib["file2"] = clone_node.getElementsByTagName("source2")[0].getAttribute("file")
            source1_node.attrib["startline1"] = clone_node.getElementsByTagName("source1")[0].getAttribute("startline")
            source2_node.attrib["startline2"] = clone_node.getElementsByTagName("source2")[0].getAttribute("startline")
            source1_node.attrib["endline1"] = clone_node.getElementsByTagName("source1")[0].getAttribute("endline")
            source2_node.attrib["endline2"] = clone_node.getElementsByTagName("source2")[0].getAttribute("endline")
            source1_node.attrib["pcid1"] = clone_node.getElementsByTagName("source1")[0].getAttribute("pcid")
            source2_node.attrib["pcid2"] = clone_node.getElementsByTagName("source2")[0].getAttribute("pcid")
            
            i += 1

    return clones

xml_doc = parse_xml(input_filename)
root = preprocess_functions(xml_doc.documentElement)

xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
with open(output_filename, "w") as f:
    f.write(xmlstr)
