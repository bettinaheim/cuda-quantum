var launchJupyterLab = (frame_id, notebook_path) => {
    let token = prompt("Enter Token:");
    let url="http://localhost:5801/lab?token="
    var frame = document.getElementById(frame_id);
    frame.src = url + token;
}
