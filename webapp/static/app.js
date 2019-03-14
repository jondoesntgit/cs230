
function SubForm(e){
	e.preventDefault();
	var url=$(this).closest('form').attr('action')
	data=$(this).closest('form').serialize();
}

/*
input.onchange = function(e){
  var sound = document.getElementById('sound');
  $.ajax({
  	type: 'POST',
  	url: '/upload_audio', 
  	data: $('#upload-form').serialize(), 
  	success: function(response) {console.log(response) }
  })
  this.files[0]
  sound.src = URL.createObjectURL(this.files[0]);
  // not really needed in this exact case, but since it is really important in other cases,
  // don't forget to revoke the blobURI when you don't need it
  sound.onend = function(e) {
    URL.revokeObjectURL(this.src);
  }
}
*/

var ws = new WebSocket('ws://localhost:5000/websocket')
ws.onopen = function() {
  ws.send('tmp.wav');
};
ws.onmessage = function (evt) {
  json = JSON.parse(evt.data)
  key = Object.keys(json)[0];
  value = Object.values(json)[0];
  switch (key) {
    case 'mel_path':
      console.log(value)
      $('#mel').attr('src', '/' + value);
      break;
    case 'vggish_path':
      $('#vggish').attr('src', '/' + value);
      break;
    case 'labels':
      value.forEach((val, i) => {
        $('#t' + i).text(val)
      })
      break;
  }
}
