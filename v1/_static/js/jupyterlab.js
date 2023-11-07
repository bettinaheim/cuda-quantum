var launchJupyterLab = (frame_id, notebook_path = null) => {
    let token = prompt("Enter Token:");
    let url="http://localhost:5801/doc/"
    var frame = document.getElementById(frame_id);
    var path = notebook_path == null ? "" : "tree/" + notebook_path;
    frame.src = url + path + "?token=" + token;
}
