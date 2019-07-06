console.log('main');

//data
var geoData = {
  'pt1': [17.936929, -98.349609],
  'pt2':[20.797201, -100.50293],
};

var commentData = {
  'pt1': 'hi',
  'pt2':'hi2',
};

//var mymap = L.map('mapid').setView([51.505, -0.09], 13);
var mymap = L.map('mapid').setView([21.330315, -100.458984], 5);

L.tileLayer('https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}', {
    attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, <a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery © <a href="https://www.mapbox.com/">Mapbox</a>',
    maxZoom: 18,
    id: 'mapbox.streets',
    accessToken: 'pk.eyJ1IjoiY2hyaXN4dDIzIiwiYSI6ImNqeHFxbWJvZjAweWMzbHA2eWhva3RlZXEifQ.Y3Hbg7CzoQikrEyfcK6HZg'
}).addTo(mymap);

//Markers
/*
var marker = L.marker([25.76032, -104.589844]).addTo(mymap);
marker.bindPopup("<b>Hello world!</b><br>I am a popup.")
*/
for(var key in geoData) {
  //var value = dict[key];
  var marker = L.marker( geoData[key]).addTo(mymap);
  marker.bindPopup(commentData[key])
  // do something with "key" and "value" variables
}


/*
var marker = L.marker([51.5, -0.09]).addTo(mymap);

var circle = L.circle([51.508, -0.11], {
    color: 'red',
    fillColor: '#f03',
    fillOpacity: 0.5,
    radius: 500
}).addTo(mymap);

var polygon = L.polygon([
    [51.509, -0.08],
    [51.503, -0.06],
    [51.51, -0.047]
]).addTo(mymap);
*/

/*
marker.bindPopup("<b>Hello world!</b><br>I am a popup.").openPopup();
circle.bindPopup("I am a circle.");
polygon.bindPopup("I am a polygon.");
*/
function onMapClick(e) {
    console.log("You clicked the map at " + e.latlng);
}

mymap.on('click', onMapClick);

//Fixed popup
/*
var popup = L.popup()
    .setLatLng([51.5, -0.09])
    .setContent("I am a standalone popup.")
    .openOn(mymap);
*/


//Popup on map with location when clicked
/*
var popup = L.popup();

function onMapClick(e) {
    popup
        .setLatLng(e.latlng)
        .setContent("You clicked the map at " + e.latlng.toString())
        .openOn(mymap);
}

mymap.on('click', onMapClick);
*/
