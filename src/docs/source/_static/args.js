function removeElementsByClassName(name){
    /*
    Remove HTML elements by class name
    */
    elems = Array.from(document.getElementsByClassName(name))
    elems.forEach(function(elem) { elem.parentNode.removeChild(elem) })
}

function addBreak(elem, class_name, after=false){
    /*
    Add a linebreak before or after an element
    */
    linebreak = document.createElement('br')
    linebreak.classList.add(...class_name.split(' '))
    if (after==true) {
        elem.parentNode.insertBefore(linebreak, elem.nextSibling)
    } else {
        elem.parentNode.insertBefore(linebreak, elem)
    }
}

function leftAlignSignatureArgs(signature_div){
    /*
    Aligns the arguemnts of a function to the left with a set margin
    */
    signature_div = signature_div.getElementsByTagName('dt')[0]
    temp_name = 'temp ' + signature_div.id.replace(/\./g, '_')
    args = Array.from(signature_div.getElementsByClassName('sig-param'))
    // Add linebreaks
    args.forEach(function(arg) {
        addBreak(arg, temp_name)
        arg.style.marginLeft = '20px'
    })
    last_arg = args[args.length-1]
    addBreak(last_arg, temp_name, after=true)
}

function parenAlignSignatureArgs(signature_div){
    /*
    Aligns the arguments of a function signature with the function's opening parenthesis.

    If one of the arguments does not fit after the parenthesis (e.g. on mobile),
    then all arguments of that function signature are separated by line breaks instead,
    and are shifted by a margin
    
    */
    signature_subdiv = signature_div.getElementsByTagName('dt')[0]
    temp_name = 'temp ' + signature_subdiv.id.replace(/\./g, '_')
    
    // Add line breaks for all arguments and calculate the maximum arg width
    args = Array.from(signature_subdiv.getElementsByClassName('sig-param'))
    args.forEach(function(arg) { addBreak(arg, temp_name) })
    args_max_width = Math.max(...args.map(arg => arg.offsetWidth))
    removeElementsByClassName(temp_name)

    // If every argument fits to the right of the parentheses
    box_right = signature_subdiv.getBoundingClientRect()['right'] - 10
    paren_right = signature_subdiv.getElementsByClassName('sig-paren')[0]
                        .getBoundingClientRect()['right'];
    if (args_max_width < box_right - paren_right) {
        // Iterate through the arguments, adding line breaks when needed
        args.forEach(function(arg) {
            arg_left = arg.getBoundingClientRect()['left']
            // If the argument is to the left of the opening parenthesis
            if (arg_left + 0.1 < paren_right) {
                addBreak(arg, temp_name)
                // Add a margin which extends to the first parenthesis
                arg_left = arg.getBoundingClientRect()['left']
                arg.style.marginLeft = paren_right - arg_left + 'px'
            }
        })
    } else {
        leftAlignSignatureArgs(signature_div)
    }
}

function resetSignatureArgs(signature_div){
    /*
    Reset the margin of each function signature argument to 0
    */
    signature_subdiv = signature_div.getElementsByTagName('dt')[0]
    temp_name = 'temp ' + signature_subdiv.id.replace(/\./g, '_')
    removeElementsByClassName(temp_name)

    args = Array.from(signature_subdiv.getElementsByClassName('sig-param'))
    args.forEach(function(arg) {
        arg.style.marginLeft = '0px'
    })
}

function alignSignatureArgs(){
    class_signatures = Array.from(document.getElementsByClassName('py class'))
    class_signatures.forEach(resetSignatureArgs)
    class_signatures.forEach(parenAlignSignatureArgs)
    
    method_signatures = Array.from(document.getElementsByClassName('py method'))
    method_signatures.forEach(resetSignatureArgs)
    method_signatures.forEach(parenAlignSignatureArgs)
}

window.addEventListener('DOMContentLoaded', alignSignatureArgs);
window.addEventListener('resize', alignSignatureArgs);