
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
