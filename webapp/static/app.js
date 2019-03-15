
chooser = document.getElementById('audioInput')
chooser.onchange = function(e) {
  var sound = document.getElementById('sound');
  sound.src = URL.createObjectURL(this.files[0]);
  // not really needed in this exact case, but since it is really important in other cases,
  // don't forget to revoke the blobURI when you don't need it
  sound.onend = function(e) {
    URL.revokeObjectURL(this.src);
  }
  $('#mel').hide()
  $('#vggish').hide()
  $('#predictions').hide()
}

form = document.getElementById('upload-form')
form.onsubmit = function(e){
	e.preventDefault();
  $('#mel').hide()
  $('#vggish').hide()
  $('#predictions').hide()
  $('#mel-spinner').show()
  $('#vggish-spinner').show()
  $('#predictions-spinner').show()
	var url=$(this).closest('form').attr('action')
	data=$(this).closest('form').serialize();
  var formData = new FormData();
  formData.append('audio', $('#audioInput')[0].files[0])
  $.ajax({
    type: 'POST',
    url: '/upload_audio', 
    data: formData, 
    processData: false,
    contentType: false,
    success: function(response) { ws.send(response) }
  })
}

/*
audioInput.onchange = function(e){
  var sound = document.getElementById('sound');
  data = $('#upload-form').closest('form').serialize()
  data = $('audioInput').serialize()
  $.ajax({
  	type: 'POST',
  	url: '/upload_audio', 
  	data: data, 
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

$('#mel').hide()
$('#vggish').hide()
$('#predictions').hide()
$('#mel-spinner').hide()
$('#vggish-spinner').hide()
$('#predictions-spinner').hide()

var ws = new WebSocket('ws://localhost:5000/websocket')
ws.onopen = function() {
  //ws.send('tmp.wav');
};
ws.onmessage = function (evt) {
  json = JSON.parse(evt.data)
  key = Object.keys(json)[0];
  value = Object.values(json)[0];
  switch (key) {
    case 'mel_path':
      $('#mel').attr('src', '/' + value);
      $('#mel').show()
      $('#mel-spinner').hide()

      break;
    case 'vggish_path':
      $('#vggish').attr('src', '/' + value);
      $('#vggish').show()
      $('#vggish-spinner').hide()
      break;
    case 'labels':
      value.forEach(function(val, i) {
        $('#t' + i).text(val)
      })
      $('#predictions').show()
      $('#predictions-spinner').hide()
      break;
  }
}
