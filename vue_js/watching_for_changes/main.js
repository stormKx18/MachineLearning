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

  //Execute code upon data changes
  //It is more optimized using computed variables
  watch:{
    counter: function(value){
      //Executes when counter changes
      var vm=this;
      setTimeout(function(){
        vm.counter=0;
      },2000); //Reset counter after two seconds
    }
  },

  methods: {
    result: function(){
      console.log('Method');
      return this.counter > 5 ? 'Greater than 5' : 'Smaller than 5'
    }
  }//End methods
}); //End Vue
