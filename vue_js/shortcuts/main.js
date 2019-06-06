new Vue({
  el: '#app',
  data: {
    link: 'http://google.com'
  },//End data
  methods:{
    changeLink: function(){
      this.link= "http://duckduckgo.com"
    }
  }
}); //End Vue
