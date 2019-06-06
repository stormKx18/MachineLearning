new Vue({
  el: '#app',
  data: {
    name: 'Chris',
    attachRed:false,
    color: 'green',
    width: 100
  },//End data
  computed:{
    divClasses: function(){
      return{
      red: this.attachRed,
      blue: !this.attachRed
    };
  },

    myStyle: function(){
      return{
        backgroundColor: this.color,
        width: this.width + 'px'
      }
    }

  }
}); //End Vue
