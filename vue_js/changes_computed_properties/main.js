new Vue({
  el: '#app',
  data: {
    counter: 0,
    secondCounter:0
  }, //End data
  //Computed: Dependent properties
  //In the html they do not use () they are threated as a property
  //It only gets called when their dependent variable are not modified
  computed:{
    output: function(){
      console.log('Computed')
      return this.counter > 5 ? 'Greater than 5' : 'Smaller than 5'
    }
  }, //End computed
  methods: {
    result: function(){
      console.log('Method');
      return this.counter > 5 ? 'Greater than 5' : 'Smaller than 5'
    }
  }//End methods
}); //End Vue
