let express = require("express");
let app = express();

app.use(function(req, res, next){
  console.log('Server running');
  next();
});

app.use(express.static("../client"));

app.listen(8081,function(){
  console.log("Serving at 8081");
})
