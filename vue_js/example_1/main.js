new Vue({
  el: '#app',
  data:{
    title:'Hello World!',
    link: 'http://google.com',
    finishedLink: '<a href="http://google.com">Google.</a>',
    counter:0,
    x:0,
    y:0,
    alertMessage:'Hi!'
  },
  methods: {
    changeTitle: function(event){
      this.title = event.target.value;
    },

    sayHello: function(){
      return 'Hello!';
    },

    increase: function(step, event){
      this.counter+=step;
    },

    updateCoordinates: function(event){
      this.x = event.clientX;
      this.y = event.clientY;
    },

    alertMe: function(event){
      this.alertMessage = event.target.value;
      alert(this.alertMessage);
    }

  } //End methods
});//End Vue
